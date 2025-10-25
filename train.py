import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# DagsHub MLflow connection
mlflow.set_tracking_uri("https://dagshub.com/<USERNAME>/<REPO>.mlflow")
mlflow.set_experiment("simple-demo")

# optional if using token
os.environ["MLFLOW_TRACKING_USERNAME"] = "<USERNAME>"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "<YOUR_DAGSHUB_TOKEN>"

# simple training
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

with mlflow.start_run():
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

print("✅ Run logged to DagsHub! Accuracy:", acc)

