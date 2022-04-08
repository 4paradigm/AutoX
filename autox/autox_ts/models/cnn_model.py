from tensorflow import keras
from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model

def cnn_model(time_varying_cols, n_steps_in, n_steps_out, used_cols, metric):
    n_features = len(time_varying_cols)

    inputs_1 = keras.Input(shape=(n_steps_in, n_features))
    x1 = keras.layers.Conv1D(50, 2, padding='causal', dilation_rate=2)(inputs_1)
    x1 = keras.layers.Conv1D(50, 2, padding='causal', dilation_rate=4)(x1)
    x1 = keras.layers.Dense(1)(x1)
    x1 = Flatten()(x1)
    cnn_output_1 = Model(inputs=inputs_1, outputs=x1)

    # Inputting Number features
    inputs_2 = Input(shape=(len(used_cols),))

    # Merging inputs
    merge = Concatenate()([cnn_output_1.output, inputs_2])
    reg_dense = Dense(128)(merge)
    out = Dense(n_steps_out, activation='linear')(reg_dense)

    # Make a model
    model = Model([cnn_output_1.input, inputs_2], out)

    # optimizer learning rate
    opt = keras.optimizers.Adam(learning_rate=0.01)

    # Compile the model
    model.compile(loss=metric, optimizer=opt, metrics=[metric])

    model.summary()

    return model
