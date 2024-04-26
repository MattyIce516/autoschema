import pytest
import pandas as pd
from unittest.mock import patch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '...', 'src')))
from autoschema.autoschema import _get_dataframe_examples
from autoschema.autoschema import auto_schema

def test_get_dataframe_examples():
    """Test that the function correctly extracts column names and example values."""
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': ['x', 'y', 'z', 'w', 'v'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    df = pd.DataFrame(data)
    
    result_df = _get_dataframe_examples(df)

    expected_data = {
        'column_name': ['A', 'B', 'C'],
        'examples': [[1, 2, 3], ['x', 'y', 'z'], [1.1, 2.2, 3.3]]
    }
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(result_df, expected_df)

def test_empty_dataframe():
    """Test the function with an empty dataframe."""
    df = pd.DataFrame()
    result_df = _get_dataframe_examples(df)
    expected_df = pd.DataFrame(columns=['column_name', 'examples'])
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_columns_with_fewer_unique_values():
    """Test the function with columns having fewer unique values than the limit of three."""
    data = {
        'A': [1, 1, 1],  # Only one unique value
        'B': ['x', 'y', None],  # Two unique values
        'C': [1.1, None, None]  # One unique value
    }
    df = pd.DataFrame(data)
    
    result_df = _get_dataframe_examples(df)

    expected_data = {
        'column_name': ['A', 'B', 'C'],
        'examples': [[1], ['x', 'y'], [1.1]]
    }
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(result_df.sort_index(axis=1), expected_df.sort_index(axis=1))

def test_auto_schema_standardization():
    """Test schema generation with and without column name standardization."""
    data = {'Name': [1, 2], 'Age': [30, 40]}
    df = pd.DataFrame(data)

    # Test without standardization
    schema_no_std = auto_schema(df, standardize_names=False, write_schema=False)
    assert list(schema_no_std['column_name']) == ['Name', 'Age']

    # Test with standardization
    schema_std = auto_schema(df, standardize_names=True, write_schema=False)
    assert list(schema_std['column_name']) == ['name', 'age']

def test_auto_schema_output_structure():
    """Test the structure of the schema DataFrame."""
    data = {'Name': [1, 2], 'Age': [30, 40]}
    df = pd.DataFrame(data)
    schema = auto_schema(df, write_schema=False)

    expected_columns = ['column_name', 'data_type', 'description', 'examples', 'default_fill_value', 'required']
    assert list(schema.columns) == expected_columns
    assert all(col in schema.columns for col in expected_columns)

@patch('pandas.DataFrame.to_excel')
def test_auto_schema_excel_output(mock_to_excel):
    """Test Excel output using mock."""
    data = {'Name': [1, 2], 'Age': [30, 40]}
    df = pd.DataFrame(data)
    auto_schema(df, write_schema=True, schema_file_name='test_schema.xlsx')

    mock_to_excel.assert_called_once_with('test_schema.xlsx', index=False)

def test_auto_schema_filename_error():
    """Test error handling for the filename."""
    data = {'Name': [1, 2], 'Age': [30, 40]}
    df = pd.DataFrame(data)

    with pytest.raises(ValueError):
        auto_schema(df, write_schema=True, schema_file_name='test_schema.txt')

