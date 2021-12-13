from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from tensorflow.keras.models import Model
from lightgbm import LGBMClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ...util.preprocess import preprocess_data_for_tree

class SurrogateTree(object):
    def __init__(self, task_type="regression", params={}, *args, **kwargs):
        """SurrogateTree
        
        Parameters
        ----------
        task_type : str, optional
            SurrogateTree type, by default "regression"
        params : dict, optional
            hyper-parameters of the Tree model, by default {}
        """
        assert task_type in ("classification", "regression")
        self.task_type = task_type
        if task_type == "classification":
            self.modelclass = DecisionTreeClassifier
        elif task_type == "regression":
            self.modelclass = DecisionTreeRegressor
        self.params = params
        self._surrogatemodel = None
    
    def surrogate(self, model, df, schema, preprocess_dict):
        """fit the SurrogateTree with the model

        Parameters
        ----------
        model : LGBMClassifier or tf.keras.models.Model
            model to be interpreted
        df : pd.DataFrame
            data used in the fitting
        schema : dict
            description of the data
        preprocess_dict : dict, optional
            preprocessors of the model, by default {}

        Returns
        -------

        """
        self.model = model
        self.schema = schema
        self.preprocess_dict = preprocess_dict
        if schema['label'] in df.columns:
            df = df.drop(schema['label'], axis=1)
            
        self._surrogatemodel = self.modelclass(**self.params)
        process_df = df.copy()
        for col in process_df.columns:
            if col in self.preprocess_dict:
                if self.preprocess_dict[col]["type"] == "Category":
                    process_df[col] = self.preprocess_dict[col]["processor"].transform(
                            process_df[col].astype(str))
                else:
                    process_df[col] = self.preprocess_dict[col]["processor"].transform(
                            process_df[[col]])

        
        if isinstance(model, Model):
            predictions  = model.predict([process_df.iloc[:, i].values
                                      for i in range(process_df.shape[1])]).flatten()
        elif isinstance(model, LGBMClassifier):
            predictions  = model.predict_proba(process_df)[:,1]
        
        if self.task_type == "classification":
            predictions = predictions > 0.5
        
        df, self.tree_preprocess_dict = preprocess_data_for_tree(df, schema)
        self._surrogatemodel.fit(df, predictions)
        self._feature_names = list(df.columns)
        tree_predictions = self._surrogatemodel.predict(df)
        r2 = 1.0 -  np.power(predictions - tree_predictions, 2).sum() / np.power(predictions - predictions.mean(), 2).sum()
        return r2
    
    def plot_tree(self, savefig=None):
        """plot the SurrogateTree

        Parameters
        ----------
        savefig : string, optional
            path to save the fig, by default None

        """
        if self._surrogatemodel is None:
            raise Exception("The surrogate model is not fitted!")
        plt.figure(dpi=300)
        plot_tree(self._surrogatemodel, filled=True, feature_names=self._feature_names)
        if savefig is None:
            plt.show()
        else:
            plt.savefig(savefig)