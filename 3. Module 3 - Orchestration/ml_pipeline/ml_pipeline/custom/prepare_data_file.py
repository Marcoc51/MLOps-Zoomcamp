import pandas as pd
import requests
import os

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@custom
def prepare_data(*args, **kwargs):
    os.makedirs("ml_pipeline/data", exist_ok=True)

    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    local_path = "ml_pipeline/data/yellow_tripdata_2023-03.parquet"

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    # ✅ Check status and save only if valid
    if response.status_code == 200:
        with open(local_path, "wb") as f:
            f.write(response.content)
        print("File downloaded successfully.")
    else:
        raise Exception(f"Failed to download file. Status: {response.status_code}\nContent: {response.text[:200]}")

    return local_path

@test
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'