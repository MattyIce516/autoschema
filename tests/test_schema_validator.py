import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from autoschema import SchemaValidator

def save_schema_rules():
    schema_rules = pd.DataFrame({
        'column_name': ['name', 'age', 'income'],
        'data_type': ['object', 'float', 'float'],
        'description': [np.nan, np.nan, np.nan],
        'examples': [['Alice', 'Bob'], [25, 30], [50000, 60000]],
        'required': [True, True, False],
        'default_fill_value': [None, 28, 0]
    })
    schema_rules.to_excel('tests/schema_rules.xlsx', index=False)

save_schema_rules()

# Sample DataFrame setup
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'name': ['Alice', None, 'Bob'],
        'age': [25, 30, None],
        'income': [50000, None, 60000]
    })

@pytest.fixture
def validator():
    # Initialize the validator with the actual Excel file
    return SchemaValidator(schema_file='tests/schema_rules.xlsx')

def test_enforce_data_type(sample_df, validator):
    validator._enforce_data_type(sample_df, 'age', 'float')
    assert sample_df['age'].dtype == float

def test_fill_missing_values(sample_df, validator):
    validator._fill_missing_values(sample_df, 'age', 28)
    assert sample_df['age'].isnull().sum() == 0

def test_validate_and_transform_column_missing_required(sample_df, validator):
    # Load the schema rules directly from the validator which has loaded them from the Excel file
    schema_rules = validator.fit(sample_df).validation_rules
    sample_df_modified = sample_df.drop(columns=['age'], errors='ignore')
    with pytest.raises(ValueError):
        validator._validate_and_transform_column(sample_df_modified, schema_rules.iloc[1])  # Assuming 'age' is the second column

def test_transform(sample_df, validator):
    transformed_df = validator.fit_transform(sample_df.copy())
    assert transformed_df['name'].isnull().sum() == 1
    assert transformed_df['age'].dtype == float
    assert transformed_df['income'].dtype == float