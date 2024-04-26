import pytest
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '...', 'src')))
from autoschema.util import standardize_column_name, standardize_dataframe_columns


@pytest.mark.parametrize("input_name, expected_output", [
    ("   Leading and Trailing   ", "leading_and_trailing"),  # Test leading/trailing spaces
    ("MixedCASE name", "mixedcase_name"),                   # Test mixed case handling
    ("name with spaces", "name_with_spaces"),               # Test spaces converted to underscores
    ("name-with-dash", "namewithdash"),                     # Test removal of dashes
    ("name, with. punctuation!", "name_with_punctuation"),  # Test removal of punctuation
    ("_name", "name"),                                      # Test leading underscore removal
    ("__name__", "name__"),                                 # Test multiple underscores handling
    ("123_number", "123_number"),                           # Test handling of digits
])
def test_standardize_column_name(input_name, expected_output):
    """Test that the standardize_column_name function works as expected."""
    assert standardize_column_name(input_name) == expected_output


def test_standardize_dataframe_columns_inplace():
    """Test that column standardization modifies the DataFrame in place."""
    data = {
        'Name 1': [1, 2],
        'Email Address.': [3, 4],
        'AGE': [5, 6]
    }
    df = pd.DataFrame(data)
    expected_column_names = ['name_1', 'email_address', 'age']

    # Standardize in place
    returned_df = standardize_dataframe_columns(df, inplace=True)

    assert df is returned_df, "The returned DataFrame should be the same instance as the input when inplace=True."
    assert list(df.columns) == expected_column_names, "Column names should be standardized correctly."

def test_standardize_dataframe_columns_not_inplace():
    """Test that column standardization returns a new DataFrame when not inplace."""
    data = {
        'Name 1': [1, 2],
        'Email Address.': [3, 4],
        'AGE': [5, 6]
    }
    df = pd.DataFrame(data)
    expected_column_names = ['name_1', 'email_address', 'age']

    # Standardize not in place
    new_df = standardize_dataframe_columns(df, inplace=False)

    assert df is not new_df, "A new DataFrame instance should be returned when inplace=False."
    assert list(df.columns) != expected_column_names, "Original DataFrame columns should not be modified."
    assert list(new_df.columns) == expected_column_names, "Column names in the new DataFrame should be standardized correctly."
