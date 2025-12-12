import pytest
from pathlib import Path
import os

def test_create_feat_imp():
    # Using PyTest assert statements to make sure figures generated from the fit_and_predict script exist
    project_root = Path(__file__).parent.parent
    feat_importance_directory = project_root / "results" / "figures" / "feature_importance.jpg"

    assert feat_importance_directory.exists(), f"File {feat_importance_directory} does not exist"