import pytest
from pathlib import Path
import os
from PIL import Image

def test_create_feat_imp():
    # Testing if genereated figure exists
    project_root = Path(__file__).parent.parent
    feat_importance_directory = project_root / "results" / "figures" / "feature_importance.jpg"

    assert feat_importance_directory.exists(), f"File {feat_importance_directory} does not exist"

    # Testing dimensions of genereated figure
    with Image.open(feat_importance_directory) as img:
        width, height = img.size
        assert width == 2480
        assert height == 1756

test_create_feat_imp()