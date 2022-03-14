import warnings
warnings.simplefilter('default')

import gc

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold,KFold

import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sklearn.preprocessing


class GatedLinearUnit(nn.Module):
    def __init__(self, input_size,
                 hidden_layer_size,
                 dropout_rate,
                 activation=None):

        super(GatedLinearUnit, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        self.W4 = torch.nn.Linear(self.input_size, self.hidden_layer_size)
        self.W5 = torch.nn.Linear(self.input_size, self.hidden_layer_size)

        if self.activation_name:
            self.activation = getattr(nn, self.activation_name)()

        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' not in n:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in n:
                torch.nn.init.zeros_(p)

    def forward(self, x):

        if self.dropout_rate:
            x = self.dropout(x)

        if self.activation_name:
            output = self.sigmoid(self.W4(x)) * self.activation(self.W5(x))
        else:
            output = self.sigmoid(self.W4(x)) * self.W5(x)

        return output


class GateAddNormNetwork(nn.Module):
    def __init__(self, input_size,
                 hidden_layer_size,
                 dropout_rate,
                 activation=None):
        super(GateAddNormNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        self.GLU = GatedLinearUnit(self.input_size,
                                   self.hidden_layer_size,
                                   self.dropout_rate,
                                   activation=self.activation_name)

        self.LayerNorm = nn.LayerNorm(self.hidden_layer_size)

    def forward(self, x, skip):
        output = self.LayerNorm(self.GLU(x) + skip)

        return output


class GatedResidualNetwork(nn.Module):
    def __init__(self,
                 hidden_layer_size,
                 input_size=None,
                 output_size=None,
                 dropout_rate=None,
                 return_gate=False):

        super(GatedResidualNetwork, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size if input_size else self.hidden_layer_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.return_gate = return_gate

        self.W1 = torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.W2 = torch.nn.Linear(self.input_size, self.hidden_layer_size)

        if self.output_size:
            self.skip_linear = torch.nn.Linear(self.input_size, self.output_size)
            self.glu_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                                                   self.output_size,
                                                   self.dropout_rate)
        else:
            self.glu_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                                                   self.hidden_layer_size,
                                                   self.dropout_rate)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if ('W2' in name or 'W3' in name) and 'bias' not in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif ('skip_linear' in name or 'W1' in name) and 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        n2 = F.elu(self.W2(x))

        n1 = self.W1(n2)

        if self.output_size:
            output = self.glu_add_norm(n1, self.skip_linear(x))
        else:
            output = self.glu_add_norm(n1, x)

        return output


class VariableSelectionNetwork(nn.Module):
    def __init__(self, hidden_layer_size,
                 dropout_rate,
                 output_size,
                 input_size=None, ):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.flattened_grn = GatedResidualNetwork(self.hidden_layer_size,
                                                  input_size=self.input_size,
                                                  output_size=self.output_size,
                                                  dropout_rate=self.dropout_rate, )

    #         self.per_feature_grn = nn.ModuleList([GatedResidualNetwork(self.hidden_layer_size,
    #                                                                    input_size = 1,
    #                                                                    output_size=1,
    #                                                                    dropout_rate=self.dropout_rate)
    #                                                       for i in range(self.output_size)])
    def forward(self, x):
        embedding = x

        flatten = torch.flatten(embedding, start_dim=1)
        mlp_outputs = self.flattened_grn(flatten)

        sparse_weights = F.softmax(mlp_outputs, dim=-1).mean(-2)
        combined = sparse_weights * flatten

        return combined, sparse_weights


def swish(x):
    return x * torch.sigmoid(x)


class SimpleMLP(nn.Module):
    def __init__(self, cat_num_classes, real_num):
        super().__init__()
        self.cat_num_classes = cat_num_classes
        self.cat_size = len(cat_num_classes)
        self.input_size = self.cat_size + real_num

        self.bn1 = nn.BatchNorm1d(real_num)
        self.output = nn.Linear(self.input_size, 1)
        self.lin_drop = nn.Dropout(0.25)
        self.sparse_weight = None
        self.temporal_vsn = VariableSelectionNetwork(hidden_layer_size=160,
                                                     input_size=self.input_size,
                                                     output_size=self.input_size,
                                                     dropout_rate=0.1)
        if cat_num_classes:
            self.build_cat_embeddings()

    def build_cat_embeddings(self):
        self.categorical_var_embeddings = nn.ModuleList(
            [nn.Embedding(self.cat_num_classes[i], 1) for i in range(len(self.cat_num_classes))])

    def forward(self, inputs):
        cat_embeddings = []
        for i in range(self.cat_size):
            e = self.categorical_var_embeddings[i](inputs['cat'][:, i])
            cat_embeddings.append(e)
        cat_embeddings = torch.cat(cat_embeddings, 1)
        real_embeddings = self.bn1(inputs['num'])
        x = torch.cat([cat_embeddings, real_embeddings], 1)
        x, self.sparse_weight = self.temporal_vsn(x)
        x = self.output(swish(x))

        return x


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device


class GRN_DATASET(Dataset):
    def __init__(self, df_data, _column_definition, mode='train'):
        self.mode = mode
        self._column_definition = _column_definition
        if _column_definition['cat']:
            self.ids = np.array(df_data.loc[:, _column_definition['cat']].values.tolist(), dtype=np.int64)
        if self._column_definition['num']:
            self.vals = np.array(df_data.loc[:, _column_definition['num']].values.tolist(), dtype=np.float32)
        if self.mode != 'test':
            self.targets = np.array(df_data.loc[:, _column_definition['target']].values, dtype=np.float64)
        self.len = df_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data_map = {}
        # print(index)
        if self._column_definition['cat']:
            data_map['cat'] = self.ids[index]
        if self._column_definition['num']:
            # print(self.vals[index])
            data_map['num'] = self.vals[index]
        if self.mode != 'test':
            targets_out = self.targets[index]
            return data_map, targets_out
        else:
            return data_map, vals_out


def train_fn(dataloaders, device, cat_num_classes, real_num):
    model = SimpleMLP(cat_num_classes, real_num).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=0.1,
                                                     patience=1,
                                                     mode='min')

    epochs = 8
    num_train_examples = len(dataloaders['train'])
    num_valid_examples = len(dataloaders['valid'])

    losses = []
    best_loss = np.inf
    weights = None
    for e in range(epochs):
        tk0 = tqdm(dataloaders['train'], total=len(dataloaders['train']))
        # train
        model.train()
        train_loss = 0
        num = 0
        for i, (maps, targets) in enumerate(tk0):
            for k, v in maps.items():
                maps[k] = v.to(device)
            # vals = vals.to(device=device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)

            yhat = model(maps)
            loss = loss_fn(yhat, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num += len(targets)
            running_loss = train_loss / num
            tk0.set_postfix(loss=running_loss)
        #             wandb.log({"Running Loss":running_loss})
        train_epoch_loss = train_loss / num_train_examples

        # valid
        model.eval()
        valid_preds = list()
        valid_loss = 0
        with torch.no_grad():
            for i, (maps, targets) in enumerate(dataloaders['valid']):
                for k, v in maps.items():
                    maps[k] = v.to(device)
                targets = targets.to(device, dtype=torch.float)

                yhat = model(maps)
                val_loss = loss_fn(yhat, targets)
                valid_loss += val_loss.item()
                valid_preds.extend(yhat.detach().cpu().numpy().flatten())
        valid_epoch_loss = valid_loss / num_valid_examples

        # change lr
        scheduler.step(valid_epoch_loss)
        # print score
        print(f"Epoch {e}, LR: {optimizer.param_groups[0]['lr']}")
        print(f"train loss: {train_epoch_loss:.8f}, valid loss {valid_epoch_loss:.8f}")
        losses.append((train_epoch_loss, valid_epoch_loss))

        # save model
        if best_loss > valid_epoch_loss:
            print(f'-- loss from {best_loss:.8f} to {valid_epoch_loss:.8f}')
            weights = model.sparse_weight.detach().cpu().numpy()
            print(f'The features weights has been recorded.')
            best_loss = valid_epoch_loss
        print()

    return weights

class Feature_selection():
    def __init__(self):
        self.new_columns = []
        #         self.new_df = None
        self._real_scaler = None
        self._cat_scalers = None
        self.weights = None
        self.selected_df = None
        self._column_definition = None
        self._num_classes_per_cat_input = None

    def fit(self, df, column_definition):
        self._column_definition = column_definition
        # 检查特征列定义和dataframe是否对应，并且转换对应数据类型,将定义的特征列取出来作为新的df,之后所有的操作都是在新的df上进行
        df = self.check_column_definition(df)
        # Scaler 和 transforme input 必须组合使用，目前来看占用时间比较多，后期可增加开关选择性使用
        self.set_scalers(df)
        df = self.transform_inputs(df)
        # 当前只在一个fold上跑，取验证得分最佳时的特征权重，后期可增加多个fold取得的权重平均
        print('Training weights\n')
        kf = KFold(n_splits=10)
        for fold_id, (trn_idx, val_idx) in enumerate(kf.split(df)):
            df_train = df.iloc[trn_idx]
            df_valid = df.iloc[val_idx]

            train_set = GRN_DATASET(df_train, self._column_definition, mode='train')
            valid_set = GRN_DATASET(df_valid, self._column_definition, mode='valid')
            dataloaders = {
                'train': DataLoader(train_set, batch_size=1024, num_workers=4, pin_memory=True, shuffle=True),
                'valid': DataLoader(valid_set, batch_size=1024, num_workers=4, pin_memory=True, shuffle=False)
            }
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.weights = np.array(
                train_fn(dataloaders, device, self._num_classes_per_cat_input, len(self._column_definition['num'])))
            print('Traning finished, please use "transform(K)" to get Top K Important Features\n')
            break

    def transform(self, df, p=10):
        ind = list(np.argpartition(self.weights, -p)[-p:])
        names = list(np.array(self.new_columns)[ind]) + self._column_definition['target']
        self.selected_df = df.loc[:, names]
        return self.selected_df

    def check_column_definition(self, df):
        print("""Checking columns' definition\n""")
        # 检查列定义是否存在
        assert 'target' and ('cat' or 'num') in self._column_definition, \
            'Lack of established columns of " target " or " num " or "cat" '
        # 检查列名是否为空
        assert self._column_definition['target'] and (self._column_definition['cat'] or self._column_definition['num']), \
            'A list with the column names cannot be empty'
        # 当前只支持一列target
        if len(self._column_definition['target']) != 1:
            self._column_definition['target'] = self._column_definition['target'][:1]
            warnings.warn("Only one column of target values is currently supported", SyntaxWarning)
        # 检查对应列名是否存在,若存在则加入到子列名列表
        for data_type in ['cat', 'num', 'target']:
            if self._column_definition[data_type]:
                for col in self._column_definition[data_type]:
                    assert col in df, f'The {data_type} column "{col}" not in dataframe'
                    self.new_columns.append(col)
        # 尝试将每一列数据类型转为对应的类型,目前默认target是连续型,考虑加上try except?
        df[self._column_definition['target'][0]] = df[self._column_definition['target'][0]].astype(float)
        if self._column_definition['num']:
            df.loc[:, self._column_definition['num']] = df.loc[:, self._column_definition['num']].apply(
                lambda row: row.astype(float))

        # #返回子数据集
        return df.loc[:, self.new_columns]

    #         self.new_df = df.loc[:,self.new_columns]

    def set_scalers(self, new_df):
        # 暂时使用MinMaxScaler，后期看需求增加其他scaler
        print('Setting scalers\n')
        self._real_scaler = MinMaxScaler(feature_range=(-1, 1))
        if "cat" in self._column_definition:
            categorical_scalers = {}
            num_classes = []
            for col in self._column_definition["cat"]:
                # Set all to str so that we don't have mixed integer/string columns
                srs = new_df[col].apply(str)
                categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
                    srs.values)
                num_classes.append(srs.nunique())
            self._cat_scalers = categorical_scalers
            self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, new_df):
        print('Scalering inputs\n')
        if "num" in self._column_definition:
            new_df[self._column_definition["num"]] = self._real_scaler.fit_transform(
                new_df[self._column_definition["num"]])

        # Format categorical inputs
        if "cat" in self._column_definition:
            for col in self._column_definition["cat"]:
                string_df = new_df[col].apply(str)
                new_df[col] = self._cat_scalers[col].transform(string_df)
        return new_df
