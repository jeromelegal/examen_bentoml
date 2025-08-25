### Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import os
import re
import bentoml
import numpy as np

### Path
raw_path = os.path.join("data", "raw/")
processed_path = os.path.join("data", "processed/")
models_path = os.path.join("data", "models/")

### Import data
df = pd.read_csv(raw_path + "admission.csv")
df = df.drop(columns=['Serial No.'])

### Class to add "rename function" in pipeline
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

### Rename columns to standardize
def rename_features(df):
    r1 = re.compile(r'^\s+|\s+$')
    r2 = re.compile(r'\s+')
    cols = df.columns.tolist()
    new_cols = []
    #print(cols)
    for col in cols:
        c = col.lower()
        c = r1.sub('', c)
        c = r2.sub('_', c)
        new_cols.append(c)
    df = df.copy()
    #print(new_cols)
    df.columns = new_cols
    return df

df = rename_features(df)

### Separe features and target
X = df.drop(columns=['chance_of_admit'])
y = df['chance_of_admit']

### Split data
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2,
                                                    random_state=42)

### Seperate features to normalize
quantitative_features = ['gre_score', 'toefl_score', 'university_rating', 'sop', 'lor', 'cgpa']
binary_features = ['research']

### Pipeline preprocessing - rename columns | normalization
preprocessing = Pipeline([
    ("rename", ColumnRenamer()),
    ("normalisation", ColumnTransformer(transformers=[
        ("quant", StandardScaler(), quantitative_features),
        ("bin", "passthrough", binary_features)
    ]))
])

# keep pandas structure
preprocessing.set_output(transform="pandas")

X_train_scaled = preprocessing.fit_transform(X_train)
X_test_scaled = preprocessing.transform(X_test)

### Save processed datasets
X_train_scaled.to_csv(processed_path + 'X_train.csv')
X_test_scaled.to_csv(processed_path + 'X_test.csv')
y_train.to_csv(processed_path + 'y_train.csv')
y_test.to_csv(processed_path + 'y_test.csv')


### Save preprocessing pipeline to add it on final pipeline
bentoml.sklearn.save_model(
    "preprocessing_pipeline",
    preprocessing,
    metadata={
        "quant_cols": quantitative_features,
        "bin_cols": binary_features,
        "feature_names_out": getattr(preprocessing, "get_feature_names_out", lambda: None)()
    }
)
