import tensorflow as tf
from collections import OrderedDict
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras import regularizers


def Dnn(inp,
        dnn_hidden_units=(256, 128, 64),
        l2_reg_dnn=0,
        seed=2048,
        dnn_dropout=0,
        dnn_activation='relu',
        dnn_use_bn=False,
        *args,
        **kwargs):
    for unit in dnn_hidden_units:
        inp = Dense(unit,
                    activation=dnn_activation,
                    kernel_regularizer=regularizers.l2(l2_reg_dnn))(inp)
        if dnn_dropout > 0:
            inp = Dropout(dnn_dropout)(inp)
        if dnn_use_bn:
            inp = BatchNormalization()(inp)
    return inp


def WideDeep(feature_columns,
             category_voc={},
             embedding_size=10,
             dnn_hidden_units=(256, 128, 64),
             l2_reg_linear=0.00001,
             l2_reg_embedding=0.00001,
             l2_reg_dnn=0,
            seed=2048,
             dnn_dropout=0,
             dnn_activation='relu',
             dnn_use_bn=False,
             task='binary',
             *args,
             **kwargs):
    if seed is not None:
        tf.random.set_seed(seed)
    input_features = []
    for fea in feature_columns:
        input_features.append(Input(shape=(1, )))
    linear_input = []
    dnn_input = []
    for i in range(len(feature_columns)):
        col = feature_columns[i]
        if col in category_voc:
            linear_input.append(Flatten()(Embedding(
                category_voc[col] + 1,
                1,
                embeddings_regularizer=regularizers.l2(l2_reg_embedding))(
                    input_features[i])))
            dnn_input.append(Flatten()(Embedding(
                category_voc[col] + 1,
                embedding_size,
                embeddings_regularizer=regularizers.l2(l2_reg_embedding))(
                    input_features[i])))
        else:
            linear_input.append(input_features[i])
            dnn_input.append(input_features[i])
    linear_input = Concatenate()(linear_input)
    dnn_input = Concatenate()(dnn_input)
    linear_logit = Dense(
        1, kernel_regularizer=regularizers.l2(l2_reg_linear))(linear_input)
    dnn_logit = Dnn(dnn_input, dnn_hidden_units, l2_reg_dnn, seed, dnn_dropout,
                    dnn_activation, dnn_use_bn)
    final_input = Concatenate()([linear_logit, dnn_logit])
    if task == 'binary':
        final_output = Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(l2_reg_linear))(final_input)
    else:
        pass
    model = tf.keras.models.Model(inputs=input_features, outputs=final_output)
    return model


def make_optimizer(name, lr, reg_l1=None, reg_l2=None, momentum=None):
    opt = None
    if name == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif name == 'lazy_adam':
        import tensorflow_addons as tfa
        opt = tfa.optimizers.LazyAdam(0.001)
    elif name == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
    elif name == 'adagrad':
        # opt = tf.keras.optimizers.AdagradOptimizer(learning_rate = lr, initial_accumulator_value=1e-10)
        opt = tf.keras.optimizers.Adagrad(learning_rate=lr)
    elif name == 'adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate=lr)
    elif name == 'rmsprop':
        opt = tf.keras.optimizers.RMSProp(learning_rate=lr)
    else:
        raise Exception("unknown optimizer {}".format(name))
    return opt