import pandas as pd
import pytest
import pandera as pa 
import os

import sys

# Make sure the scripts directory is on the import path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.validate_model import (
    validate_correct_data_format,
    validate_column_names,
    no_empty_observations,
    validate_feature_label_correlation,
    validate_feature_feature_correlation,
    validate_missingness_types_and_duplicates,
    validate_anomalies,
    validate_category_levels,
)

def make_valid_df(n=10):
    """A small valid dataframe that should pass all pandera-based validations."""
    return pd.DataFrame(
        {
            "age": [25] * n,
            "job": ["admin."] * n,
            "marital": ["single"] * n,
            "education": ["tertiary"] * n,
            "default": [False] * n,
            "balance": [100] * n,
            "housing": [True] * n,
            "loan": [False] * n,
            "contact": ["cellular"] * n,
            "day_of_week": [10] * n,
            "month": ["may"] * n,
            "duration": [120] * n,
            "campaign": [1] * n,
            "pdays": [-1] * n,
            "previous": [0] * n,
            "poutcome": ["other"] * n,
            "y": [False] * n,
        }
    )


# 1) FILE NAME FORMAT VALIDATION
def test_validate_correct_data_format_rejects_non_csv(tmp_path):
    bad_path = tmp_path / "bank_train.txt"
    bad_path.write_text("x")

    with pytest.raises(ValueError) as e:
        validate_correct_data_format(bad_path)

    assert bad_path.suffix == ".txt" 
    assert "expected '.csv'" in str(e.value) 


# 2) COLUMN NAMES VALIDATION
def test_validate_column_names_rejects_extra_columns():
    df = make_valid_df()
    df["extra_col"] = 1

    with pytest.raises(pa.errors.SchemaError) as excinfo:
        validate_column_names(df)

    msg = str(excinfo.value).lower()
    assert "extra_col" in msg
    assert "not in" in msg or "column" in msg or "schema" in msg

# 3) NO EMPTY OBSERVATIONS
def test_no_empty_observations_fails_if_any_row_all_missing():
    df = make_valid_df()
    df.loc[0, :] = pd.NA 

    with pytest.raises(Exception) as e:
        no_empty_observations(df)

    assert df.isna().all(axis=1).sum() == 1 
    assert "empty" in str(e.value).lower() or "nan" in str(e.value).lower() 


# 4) FEATURE-LABEL CORRELATION (Deepchecks)
def test_validate_feature_label_correlation_passes_on_valid_df():
    df = make_valid_df()

    # should not raise
    validate_feature_label_correlation(df)

    assert "y" in df.columns
    assert df.shape[0] > 0 


# 5) FEATURE-FEATURE CORRELATION (Deepchecks)
def test_validate_feature_label_correlation_passes_on_valid_df():
    df = make_valid_df()

    # Deepchecks requires at least two label classes
    if "y" in df.columns and len(df) >= 2:
        df.loc[0, "y"] = 0
        df.loc[1, "y"] = 1

    # should not raise
    validate_feature_label_correlation(df)


def test_validate_feature_feature_correlation_passes_on_valid_df():
    df = make_valid_df()

    # Deepchecks requires at least two label classes
    if "y" in df.columns and len(df) >= 2:
        df.loc[0, "y"] = 0
        df.loc[1, "y"] = 1

    # should not raise
    validate_feature_feature_correlation(df)



# 6) MISSINGNESS + TYPES + DUPLICATES
def test_validate_missingness_types_and_duplicates_fails_on_duplicates():
    df = make_valid_df(n=4)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True) 

    with pytest.raises(ValueError) as e:
        validate_missingness_types_and_duplicates(df)

    assert df.duplicated().any()
    assert "duplicate" in str(e.value).lower() 


# 7) OUTLIER / ANOMALY VALIDATION
def test_validate_anomalies_fails_on_out_of_range_age():
    df = make_valid_df()
    df.loc[0, "age"] = 999  

    with pytest.raises(ValueError) as e:
        validate_anomalies(df)

    assert df.loc[0, "age"] > 120  
    assert "anomaly" in str(e.value).lower() or "outlier" in str(e.value).lower() 


# 8) CATEGORY LEVELS VALIDATION
def test_validate_category_levels_fails_on_unexpected_category():
    df = make_valid_df()
    df.loc[0, "marital"] = "complicated"

    with pytest.raises(ValueError) as e:
        validate_category_levels(df)

    assert df.loc[0, "marital"] not in ["married", "single", "divorced"] 
    assert "category" in str(e.value).lower() or "level" in str(e.value).lower() 
