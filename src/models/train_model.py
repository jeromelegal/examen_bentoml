### Import librairies
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import bentoml
from sklearn.pipeline import Pipeline
from lib.custom_librarie import ColumnRenamer

### Path
processed_path = os.path.join("data", "processed/")

### Import datasets
X_train = pd.read_csv(processed_path + 'X_train.csv')
X_test = pd.read_csv(processed_path + 'X_test.csv')
y_train = pd.read_csv(processed_path + 'y_train.csv')
y_test = pd.read_csv(processed_path + 'y_test.csv')

### Model
lr = LinearRegression()

### Train model
lr.fit(X_train, y_train)

### Evaluate
y_pred_train = lr.predict(X_train)
y_pred = lr.predict(X_test)
r2_train = r2_score(y_pred_train, y_train)
r2 = r2_score(y_pred, y_test)

print(f"Le score R² train est : {r2_train}")
print(f"Le score R² test est : {r2}")

### Make final pipeline
preprocessing = bentoml.sklearn.load_model("preprocessing_pipeline:latest")
inference_model = Pipeline([
    ("preprocessing", preprocessing),
    ("linearregression", lr)
])

### Save model in Bentoml
model_ref = bentoml.sklearn.save_model("admission_lr", inference_model)
print(f"Modèle enregistré sous : {model_ref.tag}")

### Verify saving
bentoml_model_list = []
for m in bentoml.models.list():
    bentoml_model_list.append(m.tag)

if model_ref.tag in bentoml_model_list:
    print(f"Modèle 'admission_lr:latest' bien trouvé comme : {model_ref.tag}")