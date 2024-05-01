"""
The autoschema package provides tools for automated schema validation and handling of pandas DataFrames.

Modules:
- autoschema: Contains the main functionality for schema validation.
- util: Provides utility functions that support various operations within autoschema.

Usage:
    from autoschema import SchemaValidator
    validator = SchemaValidator(schema_file='path/to/schema.xlsx')

This package is designed to simplify working with data schemas in Python projects, ensuring data conforms to predefined formats and simplifying data cleaning and preprocessing tasks.

For more information, visit https://github.com/MattyIce516/autoschema.

Author: Matt Harris
License: MIT License
"""

from .util import read_universal, standardize_dataframe_columns
from .autoschema import auto_schema, SchemaValidator, ColumnSelector

