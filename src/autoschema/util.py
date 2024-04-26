"""
Utility functions for Autoschema

1. standardize_column_name(name)
2. standardize_dataframe_columns(df, inplace=True)
3. read_universal(loc, rename, **kwargs)
"""

import pathlib as pt
from typing import Union
import pandas as pd
from typeguard import typechecked
import string
import logging

logger = logging.getLogger(__name__)


@typechecked
def standardize_column_name(name: str):
    """Convert column name to a (subjectively) standardized format:
    1. Convert to lower case
    2. Strip leading and trailing white space
    3. Replace spaces with underscores
    4. Strip all other punctuation
    5. Make sure no names start with an underscore
    """
    # define the punctation that needs to be stripped
    garbage_punc = "".join([x for x in string.punctuation if x != '_'])

    name = name.translate(str.maketrans('', '', garbage_punc))
    name = name.lower().rstrip().lstrip().replace(' ', '_')
    name = name.lstrip('_')

    return name


@typechecked
def standardize_dataframe_columns(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """standardize all columns in a dataframe.

    Parameters
    -----------
    df: (pd.DataFrame) dataframe to standardize 
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
def read_universal(loc: str, rename: Union[dict, None] = None, **kwargs) -> pd.DataFrame:
    """Mulit-faceted data reading for reading from disk.
    Currently implemented extensions: .csv, .xlsx, .xls, .pkl, .parquet

    Parameters
    -----------
    loc         : (str) path to data on disk
    rename      : (dict | None) Renaming dictionary -> keys are current names, values are desired names
    **kwargs    : Any number of keyword arguments related to pandas read functions

    Returns
    ----------
    df          : (pd.DataFrame) Dataframe with the requested data

    Examples
    ----------
    >>> df = read_universal(loc="/path/to/data.csv")
    """
    sufx = pt.Path(loc).suffix

    if sufx == '.csv':
        df = pd.read_csv(loc, **kwargs)
    elif sufx in ['.xlsx', '.xls']:
        df = pd.read_excel(loc, **kwargs)
    elif sufx in ['pkl', '.pickle']:
        df = pd.read_pickle(loc, **kwargs)
    elif sufx == '.parquet':
        df = pd.read_parquet(loc, **kwargs)
    else:
        raise NotImplementedError(f"{sufx} not supported! Supported files: csv/excel/pickle/parquet")

    # Handle renaming if specified
    if (rename is not None) and (len(rename) > 0):
        df.rename(columns=rename, inplace=True)
    
    return df

