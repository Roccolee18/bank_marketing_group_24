import click
import pandas as pd
from pathlib import Path
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    FeatureLabelCorrelation,
    FeatureFeatureCorrelation,
)



## 1. FILE NAME FORMAT VALIDATION

def validate_correct_data_format(df, file_path):
    """
    Validate that the data format matches .csv.
    """

    file_path = Path(file_path)

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            f"File format is not correct: expected '.csv', got '{file_path.suffix}'"
        )

    click.echo("File format validation passed: .csv file detected.")


## 2. COLUMN NAMES VALIDATION

# Expected column names 
columns = [
    "age", "job", "marital", "education", "default",
    "balance", "housing", "loan", "contact",
    "day_of_week", "month", "duration", "campaign",
    "pdays", "previous", "poutcome", "y"
]

# Pandera schema for validating column names
column_schema = DataFrameSchema(
    columns={c: Column(nullable=True) for c in columns},
    strict=True,   # no extra columns allowed
    ordered=False  # order does not matter
)

def validate_column_names(df):
    """
    Validate that the dataset has the correct column names.
    """
    column_schema.validate(df)
    click.echo("Column name validation passed: all expected columns are present.")


## 3. NO EMPTY OBSERVATIONS

import pandera as pa


def no_empty_observations(df):
    """
    Validation to ensure no row in the dataset is entirely empty (all values missing).
    """

    # Count completely empty rows
    n_empty = df.isna().all(axis=1).sum()

    # Pandera check: no row is fully NaN
    schema = pa.DataFrameSchema(
        checks=[
            pa.Check(
                lambda df: ~(df.isna().all(axis=1)).any(),
                element_wise=False,
                error="Found one or more completely empty rows in the dataset."
            )
        ]
    )

    # Run validation
    schema.validate(df)

    print(f"No completely empty rows detected. ({n_empty} found)")


## 4. NO ANOMALOUS CORRELATION BETWEEN TARGET/RESPONSE 
## VARIABLE AND FEATURES/EXPLANATORY VARIABLE

def validate_feature_label_correlation(df):
    """
    Validate that there is no anomalously strong correlation
    between any feature and the target variable y.
    """

    bank_train_ds = Dataset(
        df,
        label="y",
        cat_features=[
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "poutcome",
        ],
    )

    check_feat_lab_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.8)
    check_feat_lab_corr_result = check_feat_lab_corr.run(dataset=bank_train_ds)

    if not check_feat_lab_corr_result.passed_conditions():
        raise ValueError("Feature-label correlation exceeds the maximum acceptable threshold.")

    click.echo("Feature-label correlation validation passed.")


## 5. NOT ANOMALOUS CORRELATIONS BETWEEN FEATURES/EXPLANATORY VARIABLES

def validate_feature_feature_correlation(df):
    """
    Validate that no pair of explanatory variables has an anomalously high correlation.
    """

    bank_train_ds = Dataset(
        df,
        label="y",
        cat_features=[
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "poutcome",
        ],
    )

    check_feat_feat_corr = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
        threshold=0.99,
    )

    check_feat_feat_corr_result = check_feat_feat_corr.run(dataset=bank_train_ds)

    if not check_feat_feat_corr_result.passed_conditions():
        raise ValueError("Feature-feature correlation exceeds the maximum acceptable threshold.")

    click.echo("Feature-feature correlation validation passed.")


## 6. MISSINGNESS NOT BEYOND THRESHOLD, CORRECT DATA TYPES IN EACH COLUMN AND NO DUPLICATE OBSERVATIONS


# Missingness thresholds per column
MISSINGNESS_THRESHOLDS = {
    "age": 0.01,
    "job": 0.50,
    "marital": 0.05,
    "education": 0.05,
    "default": 0.05,
    "balance": 0.05,
    "housing": 0.05,
    "loan": 0.05,
    "contact": 0.35,
    "day_of_week": 0.05,
    "month": 0.05,
    "duration": 0.05,
    "campaign": 0.05,
    "pdays": 0.05,
    "previous": 0.05,
    "poutcome": 0.90,
    "y": 0.05,
}

# Expected data types per column
COLUMN_TYPES = {
    "age": pa.Int,
    "job": pa.String,
    "marital": pa.String,
    "education": pa.String,
    "default": pa.Bool,
    "balance": pa.Int,
    "housing": pa.Bool,
    "loan": pa.Bool,
    "contact": pa.String,
    "day_of_week": pa.Int,
    "month": pa.String,
    "duration": pa.Int,
    "campaign": pa.Int,
    "pdays": pa.Int,
    "previous": pa.Int,
    "poutcome": pa.String,
    "y": pa.Bool,
}


def create_column_schema(col_name, col_type, threshold):
    """Create a pandera Column with a missingness check and type."""
    return Column(
        col_type,
        checks=[
            Check(
                lambda s: s.isna().mean() <= threshold,
                error=f"Missing value rate in '{col_name}' exceeds {threshold:.1%}",
                element_wise=False,
                ignore_na=False,
            )
        ],
        nullable=True,
        coerce=True,
    )


# Schema for data types + missingness thresholds
data_quality_schema = DataFrameSchema(
    {
        col_name: create_column_schema(
            col_name,
            COLUMN_TYPES[col_name],
            MISSINGNESS_THRESHOLDS[col_name],
        )
        for col_name in MISSINGNESS_THRESHOLDS.keys()
    },
    strict=True,  # only these columns allowed
    coerce=True,  # attempt to coerce types across the dataframe
)


def validate_missingness_types_and_duplicates(df):
    """
    Validate:
    - Missingness not beyond thresholds.
    - Correct data types in each column (or coercible).
    - No duplicate observations.
    """

    # Optional: print actual missingness vs threshold
    for col, threshold in MISSINGNESS_THRESHOLDS.items():
        missing_pct = df[col].isna().mean()
        click.echo(f"{col}: {missing_pct:.2%} missing (threshold: {threshold:.2%})")

    # Validate schema: types + missingness
    try:
        validated_df = data_quality_schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        raise ValueError(f"Schema validation failed: {e}") from e

    # Check for duplicate rows
    duplicates = validated_df.duplicated()
    if duplicates.any():
        num_duplicates = duplicates.sum()
        raise ValueError(f"DataFrame contains {num_duplicates} duplicate rows.")

    click.echo("Missingness, data types, and duplicate-row validation passed.")


## 7. OUTLIER/ANOMALY VALIDATION

# Outlier / Anomaly validation schema
anomaly_schema = DataFrameSchema(
    {
        "age": Column(
            int,
            checks=[
                Check.in_range(0, 120, include_min=True, include_max=True),
            ],
        ),
        "job": Column(
            str,
            checks=[
                Check(
                    lambda s: s.str.len().between(2, 50, inclusive="both"),
                    error="Job title must be between 2 and 50 characters",
                ),
            ],
            nullable=True,
        ),
        "marital": Column(
            str,
            checks=[
                Check.isin(["married", "single", "divorced"]),
            ],
        ),
        "education": Column(
            str,
            checks=[
                Check.isin(["tertiary", "secondary", "primary"]),
            ],
            nullable=True,
        ),
        "default": Column(
            str,
            checks=[
                Check.isin(["no", "yes"]),
            ],
        ),
        "balance": Column(
            int,
            checks=[
                Check.in_range(-100000, 1000000000000, include_min=True, include_max=True),
            ],
        ),
        "housing": Column(
            int,
            checks=[
                Check.in_range(0, 1, include_min=True, include_max=True),
            ],
        ),
        "loan": Column(
            int,
            checks=[
                Check.in_range(0, 1, include_min=True, include_max=True),
            ],
        ),
        "contact": Column(
            str,
            checks=[
                Check.isin(["cellular", "telephone"]),
            ],
            nullable=True,
        ),
        "day_of_week": Column(
            int,
            checks=[
                Check.in_range(1, 31, include_min=True, include_max=True),
            ],
        ),
        "month": Column(
            str,
            checks=[
                Check.isin(
                    [
                        "jan",
                        "feb",
                        "mar",
                        "apr",
                        "may",
                        "jun",
                        "jul",
                        "aug",
                        "sep",
                        "oct",
                        "nov",
                        "dec",
                    ]
                ),
            ],
        ),
        "duration": Column(
            int,
            checks=[
                Check.in_range(0, 21600, include_min=True, include_max=True),
            ],
        ),
        "campaign": Column(
            int,
            checks=[
                Check.in_range(1, 100, include_min=True, include_max=True),
            ],
        ),
        "pdays": Column(
            int,
            checks=[
                Check.in_range(-1, 10000, include_min=True, include_max=True),
            ],
        ),
        "previous": Column(
            int,
            checks=[
                Check.in_range(0, 10000000, include_min=True, include_max=True),
            ],
        ),
        "poutcome": Column(
            str,
            checks=[
                Check.isin(["failure", "other", "success"]),
            ],
            nullable=True,
        ),
        "y": Column(
            int,
            checks=[
                Check.in_range(0, 1, include_min=True, include_max=True),
            ],
        ),
    },
    strict=True,
    coerce=True,
)


def is_outlier_iqr(series, multiplier=1.5):
    """
    Helper to flag outliers in a numeric series using the IQR rule.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (series < lower_bound) | (series > upper_bound)


def validate_anomalies(df):
    """
    Validate that feature values fall within expected ranges or categories.
    """
    try:
        anomaly_schema.validate(df)
    except pa.errors.SchemaError as e:
        raise ValueError(f"Outlier/anomaly validation failed: {e}") from e

    click.echo("Outlier/anomaly validation passed.")



## 8. CORRECT CATEGORY LEVELS (I.E. NO STRING MISMATCHES OR SINGLE VALUES)

# Correct category levels (no string mismatches or unexpected values)

category_schema = DataFrameSchema(
    {
        "job": Column(
            str,
            checks=[
                Check(
                    lambda s: s.str.len().between(2, 50, inclusive="both"),
                    error="Job title must be between 2 and 50 characters",
                ),
            ],
            nullable=True,
        ),
        "marital": Column(
            str,
            checks=[
                Check.isin(["married", "single", "divorced"]),
            ],
        ),
        "education": Column(
            str,
            checks=[
                Check.isin(["tertiary", "secondary", "primary"]),
            ],
            nullable=True,
        ),
        "contact": Column(
            str,
            checks=[
                Check.isin(["cellular", "telephone"]),
            ],
            nullable=True,
        ),
        "month": Column(
            str,
            checks=[
                Check.isin(
                    [
                        "jan",
                        "feb",
                        "mar",
                        "apr",
                        "may",
                        "jun",
                        "jul",
                        "aug",
                        "sep",
                        "oct",
                        "nov",
                        "dec",
                    ]
                ),
            ],
        ),
        "poutcome": Column(
            str,
            checks=[
                Check.isin(["failure", "other", "success"]),
            ],
            nullable=True,
        ),
    },
    strict=True,
    coerce=True,
)


def validate_category_levels(df):
    """
    Validate that categorical variables have only expected category levels
    and no unexpected string values.
    """
    categorical_df = df[["job", "marital", "education", "contact", "month", "poutcome"]]

    try:
        category_schema.validate(categorical_df)
    except pa.errors.SchemaError as e:
        raise ValueError(f"Category-level validation failed: {e}") from e

    click.echo("Category-level validation passed.")


















@click.command()
@click.option(
    "--input-path",
    type=str,
    required=True,
    help="Path to the raw CSV dataset to validate.",
)
def main(input_path):
    """
    Validate dataset format and structure.
    """

    input_path = Path(input_path)

    click.echo(f"Loading dataset from: {input_path}")

    # Load the data
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Failed to load CSV file: {e}")

    click.echo("Dataset successfully loaded.")

    # VALIDATIONS 
    click.echo("Running validations...")

    validate_correct_data_format(df, input_path)
    validate_column_names(df)
    validate_missingness_types_and_duplicates(df)
    no_empty_observations(df)
    validate_feature_label_correlation(df)
    validate_feature_feature_correlation(df)
    validate_anomalies(df)
    validate_category_levels(df)


    click.echo("All validations completed successfully.")


if __name__ == "__main__":
    main()







# python scripts/validate_model.py --input-path data/raw/bank_marketing.csv
