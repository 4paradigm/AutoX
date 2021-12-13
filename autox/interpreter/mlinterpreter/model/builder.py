from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping
import numpy as np
from lightgbm import LGBMClassifier

from .nn import WideDeep, make_optimizer


def build_lgb(df, schema, params={}):
    """build a LGBMClassifier for the training data

    Parameters
    ----------
    df : pd.DataFrame
        training data
    schema : dict
        description of the Training data
    params : dict, optional
        hyperparamters of the model, by default {}
        for the details, see the doc of lightgbm 

    Returns
    -------
    LGBMClassifier
        a LGBMClassifier
    """
    feature_columns = [
        col for col in list(df.columns) if col != schema['label']
    ]
    categorial_features = [
        col['name'] for col in schema['features']
        if col['type'] == "Category" and col['name'] != schema['label']
    ]

    y = df[schema['label']]
    X = df.drop(schema['label'], axis=1)

    # split training data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # default params
    default_params = {
        "max_bin": 512,
        "learning_rate": 0.05,
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 10,
        "verbose": -1,
        "min_data": 100,
        "boost_from_average": True
    }
    for key in default_params:
        if key not in params:
            params[key] = default_params[key]
    # tune n_estimators using early_stopping
    if "n_estimators" not in params:
        lgb = LGBMClassifier(**params)
        lgb.fit(X_train,
                y_train,
                categorical_feature=categorial_features,
                early_stopping_rounds=1,
                eval_metric=['auc'],
                eval_set=((X_test, y_test)))
        best_iteration = lgb.best_iteration_
        params["n_estimators"] = int(best_iteration * 1.15)
    lgb = LGBMClassifier(**params)
    lgb.fit(X, y, categorical_feature=categorial_features)
    return lgb


def build_nn(df, schema, params={}):
    """build a wide & deep model for the training data


    Parameters
    ----------
    df : pd.DataFrame
        training data
    schema : dict
        description of the Training data
    params : dict, optional
        hyperparamters of the model, by default {}
    
            optmiter: optimizer, default is adam
            learning_rate: learning rate, default is 1e-3
            epochs: max training epochs, default is 10
            early_stopping: if use early stopping, default is True
            patience: early stopping round, default is 1
            batch_size: batch size, default is 256

    Returns
    -------
    tf.keras.models.Model
        a wide & deep model
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

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    optimizer = params.get("optmiter", "adam")
    lr = params.get("learning_rate", 1e-3)
    epochs = params.get("epochs", 10)
    batch_size = params.get("batch_size", 256)
    if params.get('early_stopping', True):
        es = EarlyStopping(
            monitor="val_auc",
            restore_best_weights=True,
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
            validation_data=([X_test[k].values for k in feature_columns], y_test),
            epochs=int(epochs),
            batch_size=batch_size,
            callbacks=[es])
        eval_metrics = history.history[es.monitor]
        best_epoch = np.argmax(eval_metrics) + 1
    else:
        best_epoch = epochs

    model = WideDeep(feature_columns=feature_columns,
                     category_voc=category_voc,
                     **params)
    model.compile(optimizer=make_optimizer(optimizer, lr),
                  loss='binary_crossentropy',
                  metrics=["AUC"])
    model.fit([X[k].values for k in feature_columns],
              y,
              batch_size=batch_size,
              epochs=best_epoch)
    return model