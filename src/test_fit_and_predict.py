import pytest
from pathlib import Path
import os


def test_fit_and_predict():
    # Using PyTest assert statements to make sure figures generated from the fit_and_predict script exist
    project_root = Path(__file__).parent.parent
    conf_matrix_directory = project_root / "results" / "figures" / "confusion_matrix.jpg"
    feat_importance_directory = project_root / "results" / "figures" / "feature_importance.jpg"

    assert conf_matrix_directory.exists(), f"File {conf_matrix_directory} does not exist"
    assert feat_importance_directory.exists(), f"File {feat_importance_directory} does not exist"
    return True