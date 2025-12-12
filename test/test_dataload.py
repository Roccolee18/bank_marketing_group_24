import os
import sys
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.data_load import load_and_save_dataset


def load_and_save_dataset(tmp_path):
    # Define output path
    output_path = tmp_path / "output"
    output_name = "test.csv"

    # Run function
    file_path, df = load_and_save_dataset(
        dataset_id=222,
        output_path=str(output_path),
        output_name=output_name
    )

    # Assertions
    assert os.path.exists(file_path)
    assert df.shape == (45211, 17)
    assert df.columns[1] == "job", f"Expected first column to be 'job', got '{df.columns[1]}'"

    # Validate written file
    loaded_df = pd.read_csv(file_path)
    assert loaded_df.equals(df)
