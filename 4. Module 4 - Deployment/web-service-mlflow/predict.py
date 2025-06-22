import pickle
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify

MLFLOW_TRACKING_URI = 'http://localhost:5000'  # Replace with your MLflow tracking URI
RUN_ID = 'c643a5050e3345c7be37ff6458851c90'  # Replace with your actual run ID
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
mlflow.set_experiment('green-taxi-experiment')

path = client.download_artifacts(run_id=RUN_ID, path='dict_vectorizer.pkl')
print(f'Downloaded dict vectorizer to {path}')

with open(path, 'rb') as f_out:
    dv = pickle.load(f_out)

logged_model = f"runs:/{RUN_ID}/model"

model = mlflow.pyfunc.load_model(logged_model)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    x = dv.transform(features)
    preds = model.predict(x)
    return float(preds[0])

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'prediction': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
