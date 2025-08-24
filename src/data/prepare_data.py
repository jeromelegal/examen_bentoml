### Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import os
import re
import bentoml

### Path
raw_path = os.path.join("..", "data", "raw/")
processed_path = os.path.join("..", "data", "processed/")
models_path = os.path.join("..", "data", "models/")

### Import data
df = pd.read_csv(raw_path + "admission.csv")
df = df.drop(columns=['Serial No.'])

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

### Normalization
# Pipeline
pipeline = ColumnTransformer(transformers=[
    ("quant", StandardScaler(), quantitative_features),
    ("bin", "passthrough", binary_features)
]) 

# keep pandas structure
pipeline.set_output(transform="pandas")

X_train_scaled = pipeline.fit_transform(X_train)
X_test_scaled = pipeline.transform(X_test)

### Save processed datasets
X_train_scaled.to_csv(processed_path + 'X_train.csv')
X_test_scaled.to_csv(processed_path + 'X_test.csv')
y_train.to_csv(processed_path + 'y_train.csv')
y_test.to_csv(processed_path + 'y_test.csv')


### Save preprocessing pipeline to add it on final pipeline
bentoml.sklearn.save_model(
    "preprocessing_pipeline",
    pipeline,
    metadata={
        "quant_cols": quantitative_features,
        "bin_cols": binary_features,
        "feature_names_out": getattr(pipeline, "get_feature_names_out", lambda: None)()
    }
)
