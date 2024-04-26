import pytest
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '...', 'src')))
from autoschema.util import read_universal

def test_read_csv():
    """Test reading CSV files."""
    # Create a temporary CSV file
    df_original = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    csv_path = 'temp_test_file.csv'
    df_original.to_csv(csv_path, index=False)
    
    # Read using the universal reader
    df_test = read_universal(csv_path)
    
    # Clean up
    os.remove(csv_path)
    
    # Check if the DataFrame read matches the original
    pd.testing.assert_frame_equal(df_original, df_test)

def test_read_excel():
    """Test reading Excel files."""
    df_original = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    excel_path = 'temp_test_file.xlsx'
    df_original.to_excel(excel_path, index=False)
    
    df_test = read_universal(excel_path)
    
    os.remove(excel_path)
    
    pd.testing.assert_frame_equal(df_original, df_test)

def test_rename_columns():
    """Test renaming functionality."""
    df_original = pd.DataFrame({'old_name': [1, 2], 'another_old_name': [3, 4]})
    csv_path = 'temp_test_file.csv'
    df_original.to_csv(csv_path, index=False)
    
    df_test = read_universal(csv_path, rename={'old_name': 'new_name', 'another_old_name': 'updated_name'})
    
    os.remove(csv_path)
    
    assert 'new_name' in df_test.columns and 'updated_name' in df_test.columns

def test_not_implemented_file_type():
    """Test error handling for unsupported file types."""
    with pytest.raises(NotImplementedError):
        read_universal('unsupported_file_type.xyz')