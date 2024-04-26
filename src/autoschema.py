import logging
import pickle
from typing import Union
import pandas as pd
import numpy as np
from typeguard import typechecked
from sklearn.base import BaseEstimator, TransformerMixin
from util import standardize_dataframe_columns, standardize_column_name

logger = logging.getLogger(__name__)


@typechecked
def _get_dataframe_examples(df: pd.DataFrame):
    """Helper function to get examples from all columns of a dataframe.
    """
    examples = {}
    for col in list(df.columns):
        values = list(df[col].dropna().unique())[:3]
        examples[col] = values
    
    example_df = pd.DataFrame(pd.Series(examples, name='examples')).reset_index(drop=False, names='column_name')
    example_df['column_name'] = example_df['column_name'].astype(str)
    return example_df


@typechecked
def auto_schema(
        df: pd.DataFrame, 
        standardize_names: bool = True,
        write_schema: bool = True,
        schema_file_name: Union[str, None] = None
    ) -> pd.DataFrame:
    """Generate schema from a dataframe.
    Convert all columns to the proper data types before calling this function.

    Parameters
    -----------
    df: (pd.DataFrame) dataframe to apply schema creation
    standardize_names: (bool) True/False whether or not to standardize column names
    write_schema: (bool) True/False whether to write out schema to excel
    schema_file_name: (str | None) name of output schema file. You can input an entire path instead of just a file name if you wish to write to a sepcific location. Defaults to schema.xlsx in the current working directory.

    Examples
    ----------
    >>> import pandas as pd
    >>> data = pd.read_csv('data.csv')
    >>> schema = auto_schema(data, write_schema=True, schema_file_name='path/to/schema.xlsx')
    """
    if standardize_names:
        df = standardize_dataframe_columns(df)
    
    # Get the example data points
    examples = _get_dataframe_examples(df)

    schema = pd.DataFrame(df.dtypes, columns=['data_type']).reset_index(drop=False, names='column_name')
    schema['description'] = np.nan
    schema = schema.merge(examples, how='left', on='column_name')
    schema['default_fill_value'] = np.nan
    schema['required'] = np.nan

    # Logic to write out schema
    if write_schema:
        if schema_file_name is None:
            schema_file_name = 'schema.xlsx'
        else:
            if not schema_file_name.endswith('.xlsx'):
                raise ValueError("'schema_file_name' must end with .xlsx")
        schema.to_excel(schema_file_name, index=False)

    return schema


@typechecked
class SchemaValidator(BaseEstimator, TransformerMixin):
    """Class to validate that a new dataset ahderes to a provided schema.

    Class can be saved and loaded as such:
    >>> validator = DataFrameValidator(schema_file='path/to/schema.xlsx')
    >>> validator.fit(some_dataframe)
    >>> 
    >>> validator.save('validator.pkl')
    >>> 
    >>> loaded_validator = DataFrameValidator.load('validator.pkl')
    >>> 
    >>> validated_data = loaded_validator.transform(another_dataframe)
    """
    def __init__(self, schema_file: str, use_standardized_names: bool = False):
        if not schema_file.endswith('.xlsx'):
            raise ValueError("'schema_file' must be an excel file with .xlsx extension!")
        self.schema_file = schema_file
        self.use_standardized_names = use_standardized_names
        self.validation_rules = None
    
    def fit(self, X, y=None):
        # Load and process the schema file
        self.validation_rules = pd.read_excel(self.schema_file)
        if self.use_standardized_names:
            self.validation_rules['column_name'] = self.validation_rules['column_name'].apply(standardize_column_name)
        return self
    
    def transform(self, X: pd.DataFrame):
        # Apply validation rules to X
        if self.use_standardized_names:
            X = standardize_dataframe_columns(X)
        
        # Apply transformations and checks based on the schema
        for _, row in self.validation_rules.iterrows():
            self._validate_and_transform_column(X, row)

        return X  # or return some validation report
    
    def _validate_and_transform_column(self, df, schema_row):
        col_name = schema_row['column_name']
        required = schema_row['required'] == True

        if col_name in df.columns:
            self._enforce_data_type(df, col_name, schema_row['data_type'])
            self._fill_missing_values(df, col_name, schema_row['default_fill_value'])
        elif required:
            raise ValueError(f"Missing required column: {col_name}")

    def _enforce_data_type(self, df, column_name, data_type):
        required_dtype = np.dtype(data_type)
        if df[column_name].dtype != required_dtype:
            df[column_name] = df[column_name].astype(required_dtype)

    def _fill_missing_values(self, df, column_name, default_value):
        if pd.notna(default_value):
            df[column_name].fillna(default_value, inplace=True)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def save(self, filename):
        """Save the instance to a file using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Load an instance from a file using pickle."""
        with open(filename, 'rb') as f:
            return pickle.load(f)


# Test functions here
if __name__ == '__main__':
    data = pd.read_csv('example.csv')
    data['column4'] = pd.to_datetime(data['column4'], dayfirst=True)
    print(data, '\n')
    
    schema = auto_schema(data, write_schema=True)
    print(schema, '\n')
