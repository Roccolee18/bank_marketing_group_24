import pytest
from pathlib import Path
import os

def test_create_conf_matrix():
    # Using PyTest assert statements to make sure figures generated from the fit_and_predict script exist
    project_root = Path(__file__).parent.parent
    conf_matrix_directory = project_root / "results" / "figures" / "confusion_matrix.jpg"

    assert conf_matrix_directory.exists(), f"File {conf_matrix_directory} does not exist"