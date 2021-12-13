from numpy.lib.npyio import save
from pandas.core import base
from pandas.core.series import Series
from pandas.io import feather_format
import shap
from tensorflow.keras.models import Model
from lightgbm import LGBMClassifier
import pandas as pd
import matplotlib.pyplot as plt

from ..util.util import basename, filename_subfix


class ShapInterpreter(object):
    def __init__(self, model, background_data, schema, preprocess_dict={}):
        """ShapInterpreter

        Parameters
        ----------
        model : LGBMClassifier or tf.keras.models.Model
            model to be interpreted
        background_data : pd.DataFrame
            background data used in the evaluation
        schema : dict
            description of the data
        preprocess_dict : dict, optional
            preprocessors of the model, by default {}

        """
        if background_data is not None:
            if schema["label"] in background_data:
                background_data = background_data.drop(schema["label"], axis=1)
        if isinstance(model, Model):
            if background_data is None:
                raise Exception(
                    "background_data must not be None for Keras Model")

            def f(X):
                return model.predict([X[:, i]
                                      for i in range(X.shape[1])]).flatten()

            self.predict = f
            self.explainer = shap.KernelExplainer(f, background_data)
            self.type = "kernel"
        elif isinstance(model, LGBMClassifier):
            self.explainer = shap.TreeExplainer(model)
            self.type = "lgb"
        self.background_data = background_data
        self.schema = schema
        self.preprocess_dict = preprocess_dict

    def _cal_shap_value(self, df, nsamples=500, use_background=True):
        df = df.copy()
        if self.schema["label"] in df:
            df.pop(self.schema["label"])
        if isinstance(df, pd.Series):
            for col in df.index:
                if col in self.preprocess_dict:
                    if self.preprocess_dict[col]["type"] == "Category":
                        df[col] = self.preprocess_dict[col][
                            "processor"].transform([str(df[col])])[0]
                    else:
                        df[col] = self.preprocess_dict[col][
                            "processor"].transform(df[col].reshape(
                                (-1, 1)))[0][0]
            df = pd.to_numeric(df)
        else:
            for col in df.columns:
                if col in self.preprocess_dict:
                    if self.preprocess_dict[col]["type"] == "Category":
                        df[col] = self.preprocess_dict[col]["processor"].transform(
                            df[col].astype(str))
                    else:
                        df[col] = self.preprocess_dict[col]["processor"].transform(
                            df[[col]])

        if self.type == "kernel":
            if isinstance(df, pd.Series):
                shap_values = self.explainer.shap_values(df, nsamples=nsamples)
            else:
                if use_background:
                    shap_values = self.explainer.shap_values(df,
                                                             nsamples=nsamples)
                else:
                    explainer = shap.KernelExplainer(self.predict, df)
                    shap_values = explainer.shap_values(df, nsamples=nsamples)
            return shap_values, df

        elif self.type == "lgb":
            if isinstance(df, pd.Series):
                df = df.to_frame().transpose()
                use_background = True
                if self.background_data is None:
                    raise Exception(
                        "for one point shap_value, the background_data is required!"
                    )
            if self.background_data is not None and use_background:
                df_all = pd.concat([df, self.background_data], axis=0)
                shap_values = self.explainer.shap_values(df_all)
            else:
                shap_values = self.explainer.shap_values(df)
            return shap_values[1][:df.shape[0]], df

    def _explainer(self, df, nsamples=500, use_background=True):
        df = df.copy()
        if self.schema["label"] in df:
            df.pop(self.schema["label"])
        if isinstance(df, pd.Series):
            for col in df.index:
                if col in self.preprocess_dict:
                    if self.preprocess_dict[col]["type"] == "Category":
                        df[col] = self.preprocess_dict[col][
                            "processor"].transform([str(df[col])])[0]
                    else:
                        df[col] = self.preprocess_dict[col][
                            "processor"].transform(df[col].reshape(
                                (-1, 1)))[0][0]
            df = pd.to_numeric(df)
        else:
            for col in df.columns:
                if col in self.preprocess_dict:
                    if self.preprocess_dict[col]["type"] == "Category":
                        df[col] = self.preprocess_dict[col]["processor"].transform(
                            df[col].astype(str))
                    else:
                        df[col] = self.preprocess_dict[col]["processor"].transform(
                            df[[col]])

        if self.type == "kernel":
            if isinstance(df, pd.Series):
                shap_values = self.explainer(df, nsamples=nsamples)
            else:
                if use_background:
                    shap_values = self.explainer(df,
                                                             nsamples=nsamples)
                else:
                    explainer = shap.KernelExplainer(self.predict, df)
                    shap_values = explainer(df, nsamples=nsamples)
            return shap_values

        elif self.type == "lgb":
            if isinstance(df, pd.Series):
                df = df.to_frame().transpose()
                use_background = True
                if self.background_data is None:
                    raise Exception(
                        "for one point shap_value, the background_data is required!"
                    )
            if self.background_data is not None and use_background:
                df_all = pd.concat([df, self.background_data], axis=0)
                shap_values = self.explainer(df_all)
            else:
                shap_values = self.explainer(df)
            return shap_values[1]

    def cal_shap_value(self, df, nsamples=500, use_background=True):
        """calculate the shap values of one point of several points

        Parameters
        ----------
        df : pd.Series or pd.DataFrame
            points to be interperted
        nsamples : int, optional
            permutation times, by default 500
        use_background : bool, optional
            if use the background data, for one point, it must be True, by default True

        Returns
        -------
        pd.DataFrame
            shap values of one point of several points
        """
        shap_value, _ = self._cal_shap_value(df,
                                          nsamples=nsamples,
                                          use_background=use_background)
        if len(shap_value.shape) == 1:
            return pd.DataFrame(shap_value.reshape(1, len(shap_value)),
                                columns=[
                                    col for col in df.index
                                    if col != self.schema['label']
                                ])
        else:
            return pd.DataFrame(shap_value,
                                columns=[
                                    col for col in df.columns
                                    if col != self.schema['label']
                                ])

    def force_plot(self, df, nsamples=500, savefig=None, use_background=True):
        """plot the shap values of one point of several points

        Parameters
        ----------
        df : pd.Series or pd.DataFrame
            points to be interperted
        nsamples : int, optional
            permutation times, by default 500
        savefig : string, optional
            path to save the fig, by default None
        use_background : bool, optional
            if use the background data, for one point, it must be True, by default True
        """
        # assert isinstance(df, pd.Series)
        if isinstance(df, pd.DataFrame):
            ins = df.shape[0]
            if ins == 1:
                self.force_plot(df.iloc[0, :],
                                nsamples=nsamples,
                                savefig=savefig,
                                use_background=use_background)
            else:
                if savefig is None:
                    for i in range(ins):
                        self.force_plot(df.iloc[i, :],
                                        nsamples=nsamples,
                                        savefig=savefig,
                                        use_background=use_background)
                else:
                    filen, subfix = filename_subfix(savefig)
                    subfix = ".{}".format(subfix) if subfix != "" else ""
                    for i in range(ins):
                        self.force_plot(df.iloc[i, :],
                                        nsamples=nsamples,
                                        savefig="{}_{}{}".format(
                                            filen, i, subfix),
                                        use_background=use_background)

        elif isinstance(df, pd.Series):
            df = df.copy()
            if self.schema["label"] in df:
                df.pop(self.schema["label"])
            shap_values, _ = self._cal_shap_value(df,
                                               nsamples,
                                               use_background=use_background)
            if savefig is not None:
                plt.clf()
            if self.type == "kernel":
                shap.force_plot(self.explainer.expected_value,
                                shap_values,
                                df,
                                matplotlib=True,
                                show=False)
            elif self.type == "lgb":
                shap.force_plot(self.explainer.expected_value[1],
                                shap_values,
                                df,
                                matplotlib=True,
                                show=False)
            if savefig is None:
                plt.show()
            if savefig is not None:
                plt.savefig(savefig, dpi=150, bbox_inches='tight')
        else:
            raise ("Unsupported input type {}".format(type(df)))

    def summary_plot(self,
                     df,
                     nsamples=500,
                     savefig=None,
                     use_background=True,
                     plot_type=None):
        """plot the summary of the shap values of the model

        Parameters
        ----------
        df : pd.DataFrame
            points to be interperted
        nsamples : int, optional
            permutation times, by default 500
        savefig : string, optional
            path to save the fig, by default None
        use_background : bool, optional
            if use the background data, for one point, it must be True, by default True
        plot_type : “dot” (default for single output), “bar” (default for multi-output), “violin”,
            or “compact_dot”. What type of summary plot to produce. Note that “compact_dot” is only used for SHAP interaction values.
        """
        if isinstance(df, pd.Series) or df.shape[0] == 1:
            self.force_plot(df,
                            nsamples,
                            savefig=savefig,
                            use_background=use_background)
        else:
            df = df.copy()
            if self.schema["label"] in df:
                df.pop(self.schema["label"])
            shap_values, _ = self._cal_shap_value(df,
                                               nsamples,
                                               use_background=use_background)
            if savefig is not None:
                plt.clf()
            shap.summary_plot(shap_values, df, show=False,  plot_type=plot_type)
            if savefig is None:
                plt.show()
            if savefig is not None:
                plt.savefig(savefig, dpi=150, bbox_inches='tight')
    
    
    def dependence_plot(self, df,
                     feature_name=None,
                     nsamples=500,
                     savefig=None,
                     use_background=True):
        assert isinstance(df, pd.DataFrame) and df.shape[0] > 1
        df = df.copy()
        if self.schema["label"] in df:
            df.pop(self.schema["label"])
        shap_values, preprocess_df = self._cal_shap_value(df,
                                            nsamples,
                                            use_background=use_background)
        if feature_name is None:
            for col in df.columns:
                shap.dependence_plot(col,
                    shap_values, preprocess_df, display_features=df)
        else:
            if isinstance(feature_name, str):
                shap.dependence_plot(feature_name,
                    shap_values, preprocess_df, display_features=df)
            elif isinstance(feature_name, (list, tuple)):
                for col in feature_name:
                    shap.dependence_plot(col,
                    shap_values, preprocess_df, display_features=df)