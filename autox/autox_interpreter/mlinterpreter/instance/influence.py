from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping
import numpy as np
from ..model.nn import WideDeep, make_optimizer
from tensorflow.keras.callbacks import Callback
import math
from tqdm import tqdm

def compute_gradient(model, X, y):
    
    data = [X.iloc[:, i].values for i in range(X.shape[1])]
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        y_pred = model(data, training=False)
        loss = model.compiled_loss(
          tf.convert_to_tensor(y), y_pred, regularization_losses=model.losses)
    return tape.gradient(loss, model.trainable_variables)

def compute_upate_u(model, X, y, u):
    
    data = [X.iloc[:, i].values for i in range(X.shape[1])]
    with tf.GradientTape() as tape2:
        tape2.watch(model.trainable_variables)
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            y_pred = model(data, training=False)
            loss = model.compiled_loss(
              tf.convert_to_tensor(y), y_pred, regularization_losses=model.losses)
            gradient = tape.gradient(loss, model.trainable_variables)
        g2 = tf.reduce_sum(tf.concat([tf.reshape(g * u,(-1,)) 
                                      for g, u in zip(gradient, u)], axis=0))
        return tape2.gradient(g2, model.trainable_variables)

def get_influential_instances_nn_sgd(df, df_display, schema, topk=10, split=0.2, params={}):
    """get topk influential instances with the algorithm in 
    [S. Hara, A. Nitanda, T. Maehara, Data Cleansing for Models Trained with SGD. 
    Advances in Neural Information Processing Systems 32 (NeurIPS'19), 2019.]

    Parameters
    ----------
    df : pd.DataFrame
        training data has been preprocessed
    df_display: pd.DataFrame
        the raw training data, which has a a one-to-one correspondence with df,
        the return DataFrame is from is from df_display
    schema : dict
        description of the Training data
    topk: int,
        number of influential instances to be selected, default is 10
    split: float,
        df will be splitted in to train and valid set with the split ratio (size of the valid set),
        and the influential metric is evaluate on the valid set, and the influential instances is selected from
        the train set.
    params : dict, optional
        hyperparamters of the model, by default {}
    
            optmiter: optimizer, default is sgd
            learning_rate: learning rate, default is 1e-3
            epochs: max training epochs, default is 10
            early_stopping: if use early stopping, default is True
            patience: early stopping round, default is 1
            batch_size: batch size, default is 256

    Returns
    -------
    pd.DataFrame
        Topk get topk influential instances in df_display
    """
    feature_columns = [
        col for col in list(df.columns) if col != schema['label']
    ]
    category_voc = {
        col['name']: df[col['name']].max()
        for col in schema['features'] if col['type'] == "Category"
    }
    y = df[schema['label']]
    X = df.drop(schema['label'], axis=1)

    assert 0 < split < 1, "split must in [0, 1]"
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=split,
                                                        random_state=42)
    weights = []
    lrs = []
    class TraceTraining(Callback):
        def on_train_batch_end(self, batch, logs=None):
            weights.append(self.model.get_weights())
            lrs.append(self.model.optimizer.learning_rate.numpy())

    optimizer = params.get("optmiter", "sgd")
    lr = params.get("learning_rate", 1e-1)
    epochs = params.get("epochs", 10)
    batch_size = params.get("batch_size", 256)

    if params.get('early_stopping', True):
        es = EarlyStopping(
            monitor="val_auc",
            restore_best_weights=False,
            patience=params.get("patience", 1),
            verbose=1,
            min_delta=0.001,
            mode="max",
            baseline=None,
        )
        model = WideDeep(feature_columns=feature_columns,
                        category_voc=category_voc,
                        **params)
        model.compile(optimizer=make_optimizer(optimizer, lr),
                    loss='binary_crossentropy',
                    metrics=["AUC"])
        weights.append(model.get_weights())
        history = model.fit(
            [X_train[k].values for k in feature_columns],
            y_train,
            shuffle=False,
            validation_data=([X_test[k].values for k in feature_columns], y_test),
            epochs=int(epochs),
            batch_size=batch_size,
            callbacks=[TraceTraining(), es])
        eval_metrics = history.history[es.monitor]
        best_epoch = np.argmax(eval_metrics) + 1
#         print(" best epoch {}".format(best_epoch))
        weights = weights[0:best_epoch * (math.ceil(len(X_train) / batch_size)) + 1]
    else:
        best_epoch = epochs
        model = WideDeep(feature_columns=feature_columns,
                         category_voc=category_voc,
                         **params)
        model.compile(optimizer=make_optimizer(optimizer, lr),
                      loss='binary_crossentropy',
                      metrics=["AUC"])
        weights.append(model.get_weights())
        model.fit(
                  [X_train[k].values for k in feature_columns],
                  y_train,
                  shuffle=False,
                  batch_size=batch_size,
                  epochs=best_epoch,
                  callbacks=[TraceTraining()],
                 )

    
    test_index = X_test.index
    train_index = X_train.index
    model.set_weights(weights[-1])
    u = [uu.numpy() for uu in compute_gradient(model, X.loc[test_index, :], y.loc[test_index])]
    batchs = np.array_split(train_index, math.ceil(len(train_index) / batch_size))
    batchs = batchs * ((len(weights) - 1) // len(batchs))
    infl = np.zeros(len(train_index) + len(test_index))
    for i in tqdm(range(len(weights) - 2, -1, -1)):
        model.set_weights(weights[i])
        lr = lrs[i]
        batch = batchs[i]
        for index in batch:
            gradient = compute_gradient(model, X.loc[[index], :], y.loc[[index]])
            infl[index] += lr * sum([(uu * g.numpy()).sum() for uu,g in zip(u, gradient)]) / len(batch)

        update_u = compute_upate_u(model, X.loc[batch, :], y.loc[batch], u)
        for j in range(len(u)):
            u[j] -= lr * update_u[j].numpy()
    
    abs_infl = np.abs(infl)
    selected_sample = (-abs_infl).argsort()[:topk]
    return df_display.loc[selected_sample, :], infl[selected_sample]

def get_influential_instances_nn(df, df_display, schema, topk=10, split=0.2, params={}):
    """get topk influential instances with the algorithm in 
    [S. Hara, A. Nitanda, T. Maehara, Data Cleansing for Models Trained with SGD. 
    Advances in Neural Information Processing Systems 32 (NeurIPS'19), 2019.]

    Parameters
    ----------
    df : pd.DataFrame
        training data has been preprocessed
    df_display: pd.DataFrame
        the raw training data, which has a a one-to-one correspondence with df,
        the return DataFrame is from is from df_display
    schema : dict
        description of the Training data
    topk: int,
        number of influential instances to be selected, default is 10
    split: float,
        df will be splitted in to train and valid set with the split ratio (size of the valid set),
        and the influential metric is evaluate on the valid set, and the influential instances is selected from
        the train set.
    params : dict, optional
        hyperparamters of the model, by default {}
    
            optmiter: optimizer, default is sgd
            learning_rate: learning rate, default is 1e-3
            epochs: max training epochs, default is 10
            early_stopping: if use early stopping, default is True
            patience: early stopping round, default is 1
            batch_size: batch size, default is 256

    Returns
    -------
    pd.DataFrame
        Topk get topk influential instances in df_display
    """
    feature_columns = [
        col for col in list(df.columns) if col != schema['label']
    ]
    category_voc = {
        col['name']: df[col['name']].max()
        for col in schema['features'] if col['type'] == "Category"
    }
    y = df[schema['label']]
    X = df.drop(schema['label'], axis=1)

    assert 0 < split < 1, "split must in [0, 1]"
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=split,
                                                        random_state=42)
    optimizer = params.get("optmiter", "adam")
    lr = params.get("learning_rate", 1e-1)
    epochs = params.get("epochs", 10)
    batch_size = params.get("batch_size", 256)

    if params.get('early_stopping', True):
        es = EarlyStopping(
            monitor="val_auc",
            restore_best_weights=False,
            patience=params.get("patience", 1),
            verbose=1,
            min_delta=0.001,
            mode="max",
            baseline=None,
        )
        model = WideDeep(feature_columns=feature_columns,
                        category_voc=category_voc,
                        **params)
        model.compile(optimizer=make_optimizer(optimizer, lr),
                    loss='binary_crossentropy',
                    metrics=["AUC"])
        history = model.fit(
            [X_train[k].values for k in feature_columns],
            y_train,
            shuffle=False,
            validation_data=([X_test[k].values for k in feature_columns], y_test),
            epochs=int(epochs),
            batch_size=batch_size,
            callbacks=[es])
        eval_metrics = history.history[es.monitor]
        best_epoch = np.argmax(eval_metrics) + 1
#         print(" best epoch {}".format(best_epoch))
    else:
        best_epoch = epochs
        model = WideDeep(feature_columns=feature_columns,
                         category_voc=category_voc,
                         **params)
        model.compile(optimizer=make_optimizer(optimizer, lr),
                      loss='binary_crossentropy',
                      metrics=["AUC"])
        model.fit(
                  [X_train[k].values for k in feature_columns],
                  y_train,
                  shuffle=False,
                  batch_size=batch_size,
                  epochs=best_epoch,
                  callbacks=None,
                 )
#     return model, X_train, X_test, y_train, y_test
    test_index = X_test.index
    train_index = X_train.index
    
    def get_inverse_hvp_lissa(v, 
                              batch_size=None,
                              scale=25, damping=0.01, num_samples=1, recursion_depth=5000):

        inverse_hvp = None
        print_iter = recursion_depth / 10

        for i in range(num_samples):
            # samples = np.random.choice(self.num_train_examples, size=recursion_depth)
            print("Runing calculate inverse hvp with lissa at iter {}/{}".format(i+1, num_samples))
            cur_estimate = [c for c in v]

            for j in tqdm(range(recursion_depth)):
                hessian_vector_val = compute_upate_u(model, 
                                                     X.loc[train_index, :], 
                                                     y.loc[train_index], 
                                                     cur_estimate)
                cur_estimate = [a + (1-damping) * b - c / scale for (a,b,c) in zip(v, cur_estimate, hessian_vector_val)] 

            if inverse_hvp is None:
                inverse_hvp = [b/scale for b in cur_estimate]
            else:
                inverse_hvp = [a + b/scale for (a, b) in zip(inverse_hvp, cur_estimate)]  

        inverse_hvp = [a/num_samples for a in inverse_hvp]
        return inverse_hvp
    
    gradient_test = [uu.numpy() for uu in compute_gradient(model, X.loc[test_index, :], y.loc[test_index])]
    inverse_hvp = get_inverse_hvp_lissa(gradient_test)
    infl = np.zeros(len(train_index) + len(test_index))
    for index in train_index:
        gradient = compute_gradient(model, X.loc[[index], :], y.loc[[index]])
        infl[index] = sum([(uu.numpy() * g.numpy()).sum() for uu,g in zip(inverse_hvp, gradient)]) / len(train_index)

    abs_infl = np.abs(infl)
    selected_sample = (-abs_infl).argsort()[:topk]
    return df_display.loc[selected_sample, :], infl[selected_sample]

