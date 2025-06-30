import os
import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def create_test_input():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 3, 0)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df_input = pd.DataFrame(data, columns=columns)

    input_file = f's3://nyc-duration/in/2023-01.parquet'
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")
    options = {
        'client_kwargs': {
            'endpoint_url': s3_endpoint_url
        }
    }

    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

def run_batch_job():
    os.system("python batch.py 2023 1")

def read_output():
    output_file = f's3://nyc-duration/out/2023-01.parquet'
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")
    options = {
        'client_kwargs': {
            'endpoint_url': s3_endpoint_url
        }
    }

    df_result = pd.read_parquet(output_file, storage_options=options)
    print("✅ Output dataframe:")
    print(df_result)
    print("✅ Sum of predicted durations:", df_result['predicted_duration'].sum())

if __name__ == "__main__":
    create_test_input()
    run_batch_job()
    read_output()
