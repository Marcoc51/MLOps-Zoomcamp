# train.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle

# Sample data
data = pd.DataFrame({
    'PULocationID': [10, 20, 10, 30],
    'DOLocationID': [50, 60, 70, 80],
    'trip_distance': [1.2, 3.4, 2.5, 4.1],
    'duration': [6.2, 15.5, 10.0, 21.1]
})

data['PU_DO'] = data['PULocationID'].astype(str) + '_' + data['DOLocationID'].astype(str)
features = data[['PU_DO', 'trip_distance']].to_dict(orient='records')
target = data['duration'].values

dv = DictVectorizer()
X = dv.fit_transform(features)
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Start MLflow run
mlflow.set_tracking_uri("http://localhost:5000")  # or another URI
mlflow.set_experiment("green-taxi-experiment")

with mlflow.start_run() as run:
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)

    # Save and log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Save and log DictVectorizer
    with open("dict_vectorizer.pkl", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact("dict_vectorizer.pkl")

    print("Run ID:", run.info.run_id)
