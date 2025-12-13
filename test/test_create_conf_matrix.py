import pytest
from pathlib import Path
import os
from PIL import Image

def test_create_conf_matrix():

    # Testing for figure existance
    project_root = Path(__file__).parent.parent
    conf_matrix_directory = project_root / "results" / "figures" / "confusion_matrix.png"

    assert conf_matrix_directory.exists(), f"File {conf_matrix_directory} does not exist"

    # Testing dimensions of figure
    with Image.open(conf_matrix_directory) as img:
        width, height = img.size
        assert width == 640
        assert height == 480