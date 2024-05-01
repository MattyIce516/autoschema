import pandas as pd
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '...', 'src')))
from autoschema.autoschema import ColumnSelector

def test_column_selector_with_required_columns():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'fare': [100, 200, 300],
        'name': ['Alice', 'Bob', 'Charlie']
    })

    # Initialize ColumnSelector with required columns
    selector = ColumnSelector(required_columns=['age', 'fare'])

    # Transform the DataFrame
    transformed_df = selector.fit_transform(df)

    # Check if the transformed DataFrame only contains the required columns
    assert list(transformed_df.columns) == ['age', 'fare'], "The DataFrame should only contain the 'age' and 'fare' columns"

def test_column_selector_missing_column():
    # Create a sample DataFrame missing one required column
    df = pd.DataFrame({
        'age': [25, 30, 35]
    })

    # Initialize ColumnSelector with a column that doesn't exist in the DataFrame
    selector = ColumnSelector(required_columns=['age', 'fare'])

    # Expect ValueError due to missing 'fare' column
    with pytest.raises(ValueError) as excinfo:
        selector.fit(df)
    assert 'fare' in str(excinfo.value), "A ValueError should be raised for the missing 'fare' column"

def test_column_selector_empty_dataframe():
    # Create an empty DataFrame
    df = pd.DataFrame()

    # Initialize ColumnSelector with any column
    selector = ColumnSelector(required_columns=['age'])

    # Expect ValueError due to missing 'age' column in an empty DataFrame
    with pytest.raises(ValueError) as excinfo:
        selector.fit(df)
    assert 'age' in str(excinfo.value), "A ValueError should be raised for the missing 'age' column"
