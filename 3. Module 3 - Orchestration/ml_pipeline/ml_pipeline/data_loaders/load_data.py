import io
import pandas as pd
import requests
from pandas import DataFrame
import pkg_resources

version = pkg_resources.get_distribution("mage-ai").version

# Question 2. Version
print("-----***-----")
print(f"Mage.ai version: {version}")

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def load_data() -> DataFrame:
    local_path = "ml_pipeline/data/yellow_tripdata_2023-03.parquet"
    df = pd.read_parquet(local_path)

    # Question 3. Creating a pipeline
    print("-----***-----")
    print(f"Number of rows: {len(df)}")

    return df


@test
def test_output(df) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
