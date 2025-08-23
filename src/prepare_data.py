### Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import os


### Path
raw_path = os.path.join("..", "data", "raw/")
processed_path = os.path.join("..", "data", "processed/")

### Import data
df = pd.read_csv(raw_path + "admission.csv")
df = df.drop(columns=['Serial No.'])

### Separe features and target
X = df.drop(columns=['Chance of Admit '])
y = df['Chance of Admit ']

### Split data
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2,
                                                    random_state=42)

### Seperate features to normalize
quantitative_features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA']
binary_features = ['Research']

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
