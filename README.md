# autoschema
`autoschema` is a Python package designed to validate and ensure that data in pandas DataFrames adheres to a predefined schema specified in an Excel file. This tool helps maintain data integrity by automating schema validations, standardizing column names, and ensuring data types and required fields are consistently upheld.

## Installation

Install `autoschema` directly from github using pip:

```bash
pip install git+https://github.com/MattyIce516/autoschema.git
```

Alternatively, you can clone the repo and install directly from the source code:

```bash
git clone https://github.com/MattyIce516/autoschema.git
cd autoschema
pip install .
```

## Features

- **Schema Validation**: Validate DataFrame columns against schema defined in an Excel file.
- **Column Standardization**: Standardize column names based on predefined rules within the schema.
- **Data Type Enforcement**: Automatically convert column data types to match those specified in the schema.
- **Missing Value Handling**: Fill missing values in DataFrame columns based on defaults specified in the schema.

## Quick Start
Here's how to quickly get started with 'autoschema':

```python
# Assuming you have a DataFrame 'data' loaded from a CSV or another source
from autoschema import auto_schema, SchemaValidator

# Create schema file from your DataFrame
schema = auto_schema(data, write_schema=True, schema_file_name='schema.xlsx')

# Initialize the validator with a path to your schema file
validator = SchemaValidator(schema_file='schema.xlsx')

# Fit the validator to your DataFrame
validator.fit(data)

# Validate and transform another DataFrame
validated_df = validator.transform(another_dataframe)
```

## Compatibility

`autoschema` is designed to seamlessly integrate with scikit-learn pipelines, enabling you to include data validation as a step in your machine learning workflows. This ensures that the data conforms to the specified schema at the model training and prediction stages, enhancing the robustness of your ML applications.

### Using `autoschema` in scikit-learn Pipelines

Here's an example of how `autoschema` can be integrated into a scikit-learn pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from autoschema import SchemaValidator

# Define the schema validator
schema_validator = SchemaValidator(schema_file='path/to/schema.xlsx')

# Create a pipeline with schema validation, feature scaling, and a classifier
pipeline = Pipeline([
    ('schema_validation', schema_validator),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Fit the pipeline on your data
pipeline.fit(X_train, y_train)

# Predict using the pipeline
y_pred = pipeline.predict(X_test)
```

## License
`autoschema` is made available under the MIT License. For more details, see the LICENSE file.

## Contact
For support or queries, please open an issue on the GitHub repository issues page.



