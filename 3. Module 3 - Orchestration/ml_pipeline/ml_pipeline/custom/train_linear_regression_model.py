from pandas import DataFrame
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def train_model(df: DataFrame, *args, **kwargs):
    """
    Trains a linear regression model on pickup/dropoff location IDs.

    Args:
        df (DataFrame): Data frame from parent block.

    Returns:
        dict: containing vectorizer and model
    """
    
    # Prepare features
    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient='records')
    target = df['duration']

    # Vectorize
    dv = DictVectorizer()
    x_train = dv.fit_transform(train_dicts)

    # Train linear regression
    lr = LinearRegression()
    lr.fit(x_train, target)

    # Print intercept
    print("-----***-----")
    print(f"Model intercept: {lr.intercept_:.3f}")

    return {
        'vectorizer': dv,
        'model': lr
    }


@test
def test_output(df) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
