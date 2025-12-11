import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src
from src.test_fit_and_predict import test_fit_and_predict

def testing_test_fit_and_predict():
    
    """Test using the imported function"""
    assert test_fit_and_predict() == True