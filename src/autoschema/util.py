"""
Utility functions for Autoschema

1. standardize_column_name(name)
2. standardize_dataframe_columns(df, inplace=True)
3. read_universal(loc, rename, **kwargs)
"""

from pathlib import Path
from typing import Union, Dict
import pandas as pd
from typeguard import typechecked
import string
import logging

logger = logging.getLogger(__name__)


@typechecked
def standardize_column_name(name: str) -> str:
    """
    Standardizes a column name by applying several transformations intended to
    conform to common naming conventions for identifiers.

    The transformations applied are:
    1. Convert to lower case.
    2. Strip leading and trailing whitespace.
    3. Replace spaces with underscores.
    4. Strip all other punctuation except underscores.
    5. Ensure the name does not start with an underscore.

    Parameters
    ----------
    name : str
        The original column name to be standardized.

    Returns
    -------
    str
        The standardized column name.

    Examples
    --------
    >>> standardize_column_name(" First Name! ")
    'first_name'
    """
    # Define the punctation that needs to be stripped
    garbage_punc = "".join([x for x in string.punctuation if x != '_'])

    # Apply transformations
    name = name.translate(str.maketrans('', '', garbage_punc))
    name = name.lower().rstrip().lstrip().replace(' ', '_')
    name = name.lstrip('_')

    return name

@typechecked
def standardize_dataframe_columns(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    Standardizes the column names of the given DataFrame according to specific naming conventions:
    1. Convert to lower case.
    2. Strip leading and trailing white space.
    3. Replace spaces with underscores.
    4. Strip all other punctuation.
    5. Ensure no names start with an underscore.

    This function can operate either in-place (modifying the original DataFrame) or return a new DataFrame
    with standardized column names, based on the 'inplace' parameter.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose columns are to be standardized.
    inplace : bool, optional
        If True, modifies the DataFrame in place and returns it. If False, returns a new DataFrame
        with modified column names, leaving the original DataFrame unchanged. Default is True.

    Returns
    -------
    pd.DataFrame
        The DataFrame with standardized column names. If 'inplace' is True, this is the same object
        as the input with modified column names. If 'inplace' is False, it is a new DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     " First Name ": [1, 2],
    ...     "Last Name!": [3, 4]
    ... })
    >>> standardize_dataframe_columns(df, inplace=False)
       first_name  last_name
    0           1          3
    1           2          4

    >>> standardize_dataframe_columns(df, inplace=True)
    >>> df
       first_name  last_name
    0           1          3
    1           2          4
    """
    new_cols = [standardize_column_name(x) for x in list(df.columns)]
    if inplace:
        df.columns = new_cols
        return df
    else:
        data = df.copy()
        data.columns = new_cols
        return data

@typechecked
def read_universal(loc: str, rename: Union[Dict[str, str], None] = None, **kwargs) -> pd.DataFrame:
    """
    Reads data from various file formats into a pandas DataFrame, allowing optional renaming of columns.

    Supported file extensions: .csv, .xlsx, .xls, .pkl, .parquet.

    Parameters
    ----------
    loc : str
        Path to the data file on disk.
    rename : dict of str: str, optional
        A dictionary mapping existing column names to new column names.
    **kwargs :
        Additional keyword arguments to pass to the underlying pandas read function.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the loaded data.

    Raises
    ------
    NotImplementedError
        If the file extension is not supported.

    Examples
    --------
    >>> df = read_universal("/path/to/data.csv")
    >>> df = read_universal("/path/to/data.xlsx", rename={'old_name': 'new_name'})
    """
    suffix = Path(loc).suffix

    if suffix == '.csv':
        df = pd.read_csv(loc, **kwargs)
    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(loc, **kwargs)
    elif suffix in ['.pkl', '.pickle']:
        df = pd.read_pickle(loc, **kwargs)
    elif suffix == '.parquet':
        df = pd.read_parquet(loc, **kwargs)
    else:
        raise NotImplementedError(f"File extension '{suffix}' is not supported. Supported extensions are: csv, xlsx, xls, pkl, parquet.")

    # Handle renaming if specified
    if rename:
        df.rename(columns=rename, inplace=True)
    
    return df

