#!/usr/bin/env python
# coding: utf-8

import os
import sys
import uuid
import pickle
import pandas as pd
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids

def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    df['ride_id'] = generate_uuids(len(df))

    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts

def load_model(run_id):
    mlflow.set_tracking_uri("http://localhost:5000")
    logged_model = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(logged_model)
    
    # Download and load the DictVectorizer
    client = mlflow.tracking.MlflowClient()
    dv_path = client.download_artifacts(run_id, "dict_vectorizer.pkl")
    with open(dv_path, "rb") as f_in:
        dv = pickle.load(f_in)

    return model, dv


def apply_model(input_file, run_id, output_file):

    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    model, dv = load_model(run_id)

    x = dv.transform(dicts)  # ✅ Apply the same transformation used in training
    y_pred = model.predict(x)

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id
    
    df_result.to_parquet(output_file, index=False)

def run():
    taxi_type = sys.argv[1] # 'green'
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) # 3

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    RUN_ID = os.getenv('RUN_ID', '52ca2d6c02dc4733b4bbed474b4e3a4b')

    apply_model(
        input_file=input_file, 
        run_id=RUN_ID, 
        output_file=output_file
    )

if __name__ == "__main__":
    run()
