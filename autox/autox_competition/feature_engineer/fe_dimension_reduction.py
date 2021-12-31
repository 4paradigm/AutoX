import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

class FeatureDimensionReduction:
    def __init__(self):
        self.id_column = None
        self.target = None
        self.silence_cols = []
        self.used_features = []
        self.n_comp = None

    def fit(self, df, id_column, target, silence_cols=[], n_comp = 12):

        self.id_column = id_column
        self.target = target
        self.silence_cols = silence_cols
        self.n_comp = n_comp

        train = df[~df[target].isnull()]
        test = df[df[target].isnull()]

        train.drop(id_column + [target], axis=1, inplace=True)
        test.drop(id_column + [target], axis=1, inplace=True)

        used_features = test.describe().columns
        used_features = [x for x in used_features if x not in silence_cols]
        self.used_features = used_features

        # tSVD
        # self.tsvd = TruncatedSVD(n_components=self.n_comp, random_state=420)
        # self.tsvd.fit(train[self.used_features].fillna(0))

        # PCA
        self.pca = PCA(n_components=self.n_comp, random_state=420, svd_solver='full')
        self.pca.fit(train[self.used_features].fillna(0))

        # ICA
        self.ica = FastICA(n_components=self.n_comp, random_state=420)
        self.ica.fit(train[self.used_features].fillna(0))

        # GRP
        self.grp = GaussianRandomProjection(n_components=self.n_comp, eps=0.1, random_state=420)
        self.grp.fit(train[self.used_features].fillna(0))

        # SRP
        self.srp = SparseRandomProjection(n_components=self.n_comp, dense_output=True, random_state=420)
        self.srp.fit(train[self.used_features].fillna(0))

    def transform(self, df):
        result = pd.DataFrame()
        # tsvd_results = self.tsvd.transform(df[self.used_features])
        pca_results  = self.pca.transform(df[self.used_features])
        ica_results  = self.ica.transform(df[self.used_features])
        grp_results  = self.grp.transform(df[self.used_features])
        srp_results  = self.srp.transform(df[self.used_features])

        for i in range(1, self.n_comp + 1):
            # result['tsvd_' + str(i)] = tsvd_results[:, i - 1]
            result['pca_' + str(i)] = pca_results[:, i - 1]
            result['ica_' + str(i)] = ica_results[:, i - 1]
            result['grp_' + str(i)] = grp_results[:, i - 1]
            result['srp_' + str(i)] = srp_results[:, i - 1]

        return result

    def fit_transform(self, df, id_column, target, silence_cols=[], n_comp = 12):
        self.fit(df, id_column, target, silence_cols=silence_cols, n_comp = n_comp)
        return self.transform(df)