import os
import sys
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

# Make sure the scripts directory is on the import path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.split_n_preprocess import load_and_clean, split_data, build_preprocessor

def test_load_and_clean(tmp_path):
    """
    Ensure load_and_clean:
    - removes rows with 'unknown'
    - maps yes/no fields into numeric
    """

    df = pd.DataFrame({
        "education": ["primary", "unknown"],
        "job": ["admin.", "technician"],
        "marital": ["single", "unknown"],
        "housing": ["yes", "no"],
        "loan": ["no", "yes"],
        "y": ["yes", "no"],
    })

    raw_file = tmp_path / "raw.csv"
    df.to_csv(raw_file, index=False)

    cleaned = load_and_clean(raw_file)

    assert cleaned.shape[0] == 1

    row = cleaned.iloc[0]
    assert row["y"] == 1
    assert row["housing"] == 1
    assert row["loan"] == 0

def test_build_preprocessor():
    """
    Ensure build_preprocessor:
    - returns a ColumnTransformer
    - includes numeric and categorical transformers
    """

    df = pd.DataFrame({
        "age": [30, 40],
        "balance": [1000, 1500],
        "job": ["admin.", "services"],
        "y": [1, 0],
    })

    preprocessor = build_preprocessor(df)

    assert isinstance(preprocessor, ColumnTransformer)

    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names

def test_split_data_stratification():
    """
    Ensure split_data:
    - produces 70/30 split
    """

    df = pd.DataFrame({
        "age": list(range(100)),
        "y":   [1]*30 + [0]*70
    })

    train_df, test_df = split_data(df, seed=42)

    # Assert the lengths
    assert len(train_df) == 70
    assert len(test_df) == 30