import logging
import pickle
from typing import Union
import pandas as pd
import numpy as np
from typeguard import typechecked
from sklearn.base import BaseEstimator, TransformerMixin
from .util import standardize_dataframe_columns, standardize_column_name

logger = logging.getLogger(__name__)


@typechecked
def _get_dataframe_examples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts up to three unique non-null examples from each column of the input DataFrame 
    and returns a new DataFrame with these examples. Each column in the new DataFrame represents 
    a column from the original DataFrame, with corresponding examples listed in rows.

    Ensures that column names are converted to strings to avoid type mismatches in subsequent processing.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to extract examples.

    Returns
    -------
    pd.DataFrame
        A new DataFrame where each row corresponds to a column in the original DataFrame. 
        It contains two columns:
        - 'column_name': the name of the column in the original DataFrame, converted to string.
        - 'examples': a list containing up to three unique examples from the column.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "A": [1, 2, 2, 4],
    ...     "B": ["x", "y", "y", "z"]
    ... })
    >>> example_df = _get_dataframe_examples(df)
    >>> print(example_df)
      column_name    examples
    0           A  [1, 2, 4]
    1           B  ['x', 'y', 'z']
    """
    examples = {}
    for col in df.columns:
        # Extract up to three unique non-null values
        values = list(df[col].dropna().unique())[:3]
        examples[col] = values
    
    # Create a DataFrame to display the examples
    example_df = pd.DataFrame({
        "column_name": [str(col) for col in examples.keys()],  # Convert column names to strings
        "examples": examples.values()
    })

    return example_df

@typechecked
def auto_schema(
        df: pd.DataFrame, 
        standardize_names: bool = True,
        write_schema: bool = True,
        schema_file_name: Union[str, None] = None
    ) -> pd.DataFrame:
    """
    Analyzes a DataFrame and generates a schema describing its structure, including data types
    and examples from each column. Optionally, it can standardize column names and write the schema 
    to an Excel file. 

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to generate the schema.
    standardize_names : bool, optional
        If True, standardizes column names before generating the schema. Default is True.
    write_schema : bool, optional
        If True, writes the generated schema to an Excel file. Default is True.
    schema_file_name : str or None, optional
        The file path including the file name where the schema should be written. If not specified,
        the schema will be written to 'schema.xlsx' in the current working directory. The file name
        must end with '.xlsx'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the schema of the input DataFrame. The schema includes the column names,
        data types, example values, and placeholders for descriptions, default fill values and boolean indicator 
        for whether or not the column is required.

    Raises
    ------
    ValueError
        If the `schema_file_name` does not end with '.xlsx' when `write_schema` is True.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.read_csv('data.csv')
    >>> print(data.head())
        id    name
    0   1     Alice
    1   2     Bob
    2   3     Charlie
    >>> schema = auto_schema(data, write_schema=True, schema_file_name='path/to/schema.xlsx')
    >>> print(schema)
        column_name    data_type    examples    description    default_fill_value    required
    0   id              int64        [1, 2, 3]   NaN             NaN                   NaN
    1   name            object       [Alice, Bob, Charlie] NaN   NaN                   NaN
    """
    if standardize_names:
        df = standardize_dataframe_columns(df)
    
    # Get the example data points
    examples = _get_dataframe_examples(df)

    # Create schema frame
    schema = pd.DataFrame(df.dtypes, columns=['data_type']).reset_index(drop=False, names=['column_name'])
    schema['description'] = np.nan
    schema['default_fill_value'] = np.nan
    schema['required'] = np.nan
    schema = schema.merge(examples, how='left', on='column_name')

    # Logic to write out schema
    if write_schema:
        if schema_file_name is None:
            schema_file_name = 'schema.xlsx'
        elif not schema_file_name.endswith('.xlsx'):
            raise ValueError("'schema_file_name' must end with .xlsx")
        schema.to_excel(schema_file_name, index=False)

    return schema

@typechecked
class SchemaValidator(BaseEstimator, TransformerMixin):
    """
    A class for validating that new datasets adhere to a provided schema defined in an Excel file.
    This validator can standardize column names based on the schema and ensure that the data
    in a DataFrame conforms to the specified data types and required constraints.

    The validator can be saved to and loaded from a pickle file, allowing for reuse across different
    sessions and datasets.

    Parameters
    ----------
    schema_file : str
        Path to an Excel file (.xlsx) containing the schema definition.
    use_standardized_names : bool, optional
        If True, standardizes column names in the input DataFrame based on the schema before validation.
        Default is False.

    Raises
    ------
    ValueError
        If the `schema_file` does not end with '.xlsx', indicating the file format is not supported.

    Examples
    --------
    >>> validator = SchemaValidator(schema_file='path/to/schema.xlsx')
    >>> validator.fit(some_dataframe)
    >>> validator.save('validator.pkl')
    >>> loaded_validator = SchemaValidator.load('validator.pkl')
    >>> validated_data = loaded_validator.transform(another_dataframe)
    """
    def __init__(self, schema_file: str, use_standardized_names: bool = False):
        if not schema_file.endswith('.xlsx'):
            raise ValueError("'schema_file' must be an excel file with .xlsx extension!")
        self.schema_file = schema_file
        self.use_standardized_names = use_standardized_names
        self.validation_rules = None
    
    def fit(self, X, y=None):
        """
        Loads the schema from the specified Excel file and prepares the validator.
        Optionally standardizes column names according to the schema.

        Parameters
        ----------
        X : pd.DataFrame
            The DataFrame used to fit the model. Not modified.
        y : None
            Unused parameter for compatibility with sklearn's TransformerMixin.

        Returns
        -------
        self : SchemaValidator
            The instance itself.
        """
        self.validation_rules = pd.read_excel(self.schema_file)
        if self.use_standardized_names:
            self.validation_rules['column_name'] = self.validation_rules['column_name'].apply(standardize_column_name)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the DataFrame against the schema, applying data type conversions and filling
        missing values as specified.

        Parameters
        ----------
        X : pd.DataFrame
            The DataFrame to validate and transform.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame which conforms to the schema.
        """
        if self.use_standardized_names:
            X = standardize_dataframe_columns(X)
        
        # Apply transformations and checks based on the schema
        for _, row in self.validation_rules.iterrows():
            self._validate_and_transform_column(X, row)

        return X
    
    def fit_transform(self, X, y=None):
        """
        Fits the validator to the DataFrame and then transforms it, ensuring it adheres to the schema.

        Parameters
        ----------
        X : pd.DataFrame
            The DataFrame to fit and transform.
        y : None
            Unused parameter for compatibility with sklearn's TransformerMixin.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame which conforms to the schema.
        """
        return self.fit(X, y).transform(X)
    
    def _validate_and_transform_column(self, df, schema_row):
        """
        Validates and transforms a column in the DataFrame according to the schema rules provided in schema_row.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame that is being validated and transformed.
        schema_row : pd.Series
            A Series containing the validation rules for a single column, including the name of the column,
            whether it is required, the data type it should have, and the default fill value.

        Raises
        ------
        ValueError
            If a required column is missing from the DataFrame.

        This method applies data type enforcement and fills missing values as specified in the schema.
        """
        col_name = schema_row['column_name']
        required = schema_row['required'] == True

        if col_name in df.columns:
            self._enforce_data_type(df, col_name, schema_row['data_type'])
            self._fill_missing_values(df, col_name, schema_row['default_fill_value'])
        elif required:
            raise ValueError(f"Missing required column: {col_name}")

    def _enforce_data_type(self, df, column_name, data_type):
        """
        Enforces the data type of a specified column in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame whose column data type is being enforced.
        column_name : str
            The name of the column for which the data type should be enforced.
        data_type : str
            The desired data type for the column as a string (e.g., 'int64', 'float32').

        This method converts the column to the specified data type.
        """
        required_dtype = np.dtype(data_type)
        if df[column_name].dtype != required_dtype:
            df[column_name] = df[column_name].astype(required_dtype)

    def _fill_missing_values(self, df, column_name, default_value):
        """
        Fills missing values in a specified column of the DataFrame using a default value.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame whose missing values are being filled.
        column_name : str
            The name of the column for which missing values should be filled.
        default_value
            The value to use for filling missing values in the column.

        If the default_value is not None, this method will fill all missing values in the column with it.
        """
        if pd.notna(default_value):
            df[column_name] = df[column_name].fillna(default_value)

    def save(self, filename):
        """
        Save the validator instance to a file using pickle.

        Parameters
        ----------
        filename : str
            The name of the file where the instance should be saved.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """
        Load a SchemaValidator instance from a pickle file.

        Parameters
        ----------
        filename : str
            The name of the file to load the instance from.

        Returns
        -------
        SchemaValidator
            The loaded SchemaValidator instance.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    A transformer class for selecting specific columns from a DataFrame.

    This is useful in a pipeline where only certain features are needed for modeling and
    can help ensure that the DataFrame passed through the pipeline only contains
    the required fields.

    Parameters:
    ----------
    required_columns : list
        A list of column names to retain in the DataFrame.

    Example:
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> pipeline = Pipeline([
    ...     ('selector', ColumnSelector(required_columns=['age', 'fare'])),
    ...     ('classifier', RandomForestClassifier())
    ... ])
    >>> pipeline.fit(X_train, y_train)  # X_train must include at least 'age' and 'fare'
    """

    def __init__(self, required_columns):
        self.required_columns = required_columns

    def fit(self, X, y=None):
        # Check if all the required columns are in the DataFrame
        missing_cols = [col for col in self.required_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"The following required columns are missing: {missing_cols}")
        return self

    def transform(self, X):
        """
        Transforms the DataFrame by retaining only the required columns.

        Parameters:
        ----------
        X : pd.DataFrame
            Input DataFrame to transform.

        Returns:
        -------
        pd.DataFrame
            Transformed DataFrame containing only the required columns.
        """
        return X[self.required_columns].copy()

    def fit_transform(self, X, y=None):
        """
        Fits to the data, then transforms it. This is a shorthand for fit(X, y).transform(X).

        Parameters:
        ----------
        X : pd.DataFrame
            Input DataFrame to fit and transform.
        y : None
            Unused parameter, included for compatibility with Pipeline.

        Returns:
        -------
        pd.DataFrame
            Transformed DataFrame containing only the required columns.
        """
        # First, fit to get any necessary validation done
        self.fit(X, y)
        # Now transform and return the transformed data
        return self.transform(X)


