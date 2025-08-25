from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import re
import numpy as np

class ColumnRenamer(BaseEstimator, TransformerMixin):
    """ Class to integrade rename function to pipeline """
    def __init__(self):
        self.r1 = re.compile(r'^\s+|\s+$')
        self.r2 = re.compile(r'\s+')
        self.mapping = None

    def new_name(self, col: str) -> str:
        c = col.lower()
        c = self.r1.sub('', c)
        c = self.r2.sub('_', c)
        return c
    
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnRenamer waits for pandas dataframe.")
        self.mapping = {c: self.new_name(c) for c in X.columns}
        return self
    
    def transform(self, X):
        return X.rename(columns=self.mapping, copy=True)

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.mapping.values()))