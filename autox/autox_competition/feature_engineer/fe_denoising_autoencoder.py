import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.layers import Input, Dense,BatchNormalization,Dropout
from keras.models import Model
from keras.regularizers import l2

class FeatureDenoisingAutoencoder:
    def __init__(self):
        self.id_column = None
        self.target = None
        self.silence_cols = []
        self.used_features = []
        self.n_comp = None
        self.feature_type = None

    def fit(self, df, id_column, target, feature_type, silence_cols=[], n_comp = 12):

        self.id_column = id_column
        self.target = target
        self.silence_cols = silence_cols
        self.n_comp = n_comp
        self.feature_type = feature_type

        shape_of_train = df[~df[target].isnull()].shape[0]
        dataset = df.copy()
        dataset.drop(id_column + [target], axis=1, inplace=True)

        used_features = dataset.describe().columns
        used_features = [x for x in used_features if x not in silence_cols]
        self.used_features = used_features

        cat_vars = [x for x in used_features if feature_type[x] == 'cat']
        for c in cat_vars:
            t_data = pd.get_dummies(dataset[c], prefix=c)
            dataset = pd.concat([dataset, t_data], axis=1)
        dataset.drop(cat_vars, axis=1, inplace=True)

        self.sc = StandardScaler()
        self.sc.fit(dataset)
        dataset = self.sc.transform(dataset)
        dataset = dataset + 0.0001 * np.random.normal(loc=0.0, scale=1.0, size=dataset.shape)

        train = dataset[:shape_of_train]
        test = dataset[shape_of_train:]

        l2_reg_embedding = 1e-5
        init_dim = train.shape[1]

        input_row = Input(shape=(init_dim,))
        encoded = Dense(512, activation='elu', kernel_regularizer=l2(l2_reg_embedding))(input_row)
        encoded = Dropout(0.2)(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Dense(256, activation='elu', kernel_regularizer=l2(l2_reg_embedding))(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(128, activation='elu', kernel_regularizer=l2(l2_reg_embedding))(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(64, activation='elu', kernel_regularizer=l2(l2_reg_embedding))(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(32, activation='elu', kernel_regularizer=l2(l2_reg_embedding))(encoded)
        encoded = Dense(n_comp, activation='elu')(encoded)

        decoded = Dense(32, activation='elu', kernel_regularizer=l2(l2_reg_embedding))(encoded)
        encoded = Dropout(0.2)(encoded)
        decoded = Dense(64, activation='elu', kernel_regularizer=l2(l2_reg_embedding))(decoded)
        encoded = Dropout(0.2)(encoded)
        decoded = Dense(128, activation='elu', kernel_regularizer=l2(l2_reg_embedding))(decoded)
        encoded = Dropout(0.2)(encoded)
        decoded = Dense(256, activation='elu', kernel_regularizer=l2(l2_reg_embedding))(decoded)
        encoded = Dropout(0.2)(encoded)
        encoded = BatchNormalization()(encoded)
        decoded = Dense(512, activation='elu', kernel_regularizer=l2(l2_reg_embedding))(decoded)
        decoded = Dense(init_dim, activation='sigmoid')(decoded)

        self.autoencoder = Model(inputs=input_row, outputs=decoded)
        self.autoencoder.compile(optimizer='rmsprop', loss='mse')
        self.autoencoder.fit(train, train,
                        batch_size=512,
                        shuffle=True, validation_data=(test, test), epochs=3)

        # compressing the data
        self.encoder = Model(inputs=input_row, outputs=encoded)


    def transform(self, df):
        result = pd.DataFrame()

        dataset = df.copy()
        dataset.drop(self.id_column + [self.target], axis=1, inplace=True)

        used_features = df.describe().columns
        used_features = [x for x in used_features if x not in self.silence_cols]
        self.used_features = used_features

        cat_vars = [x for x in used_features if self.feature_type[x] == 'cat']
        for c in cat_vars:
            t_data = pd.get_dummies(dataset[c], prefix=c)
            dataset = pd.concat([dataset, t_data], axis=1)
        dataset.drop(cat_vars, axis=1, inplace=True)
        dataset = self.sc.transform(dataset)
        # dataset = dataset + 0.0001 * np.random.normal(loc=0.0, scale=1.0, size=dataset.shape)

        df_compress = self.encoder.predict(dataset)
        for j in range(df_compress.shape[1]):
            result['denoising_auto_encoder_' + str(j+1)] = df_compress[:, j]

        return result

    def fit_transform(self, df, id_column, target, feature_type, silence_cols=[], n_comp = 12):
        self.fit(df, id_column, target, feature_type=feature_type, silence_cols=silence_cols, n_comp = n_comp)
        return self.transform(df)