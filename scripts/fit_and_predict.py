import os
import click
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, 
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from pathlib import Path
import sys
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import LabelDrift, MultivariateDrift

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.validation_after_splitting import validation_after_splitting
from src.create_conf_matrix import create_conf_matrix
from src.create_feat_imp import create_feat_imp

@click.command()
@click.option('--save_location', type=str, required=True, help="Directory to save generated artifacts to")
@click.option('--preprocessor_pickle', type=str, required=True, help="Directory to pickle file of preprocessor pipeline from previous script")
@click.option('--train_dataset_path', type=str, required=True, help="Directory to access training dataset as CSVs")
@click.option('--test_dataset_path', type=str, required=True, help="Directory to accesstesting dataset as CSVs")

def main(save_location, preprocessor_pickle, train_dataset_path, test_dataset_path):
    """
    Fitting the model.
    """
    # Reading in pickle file from previous step
    with open(preprocessor_pickle, 'rb') as f:
        preprocessor = pickle.load(f)

    # Reading in CSVs of training and testing datasets from previous step
    training_dataset = pd.read_csv(train_dataset_path)
    test_dataset = pd.read_csv(test_dataset_path)
    X_train = training_dataset.drop("y", axis = 1)
    y_train = training_dataset["y"]
    X_test = test_dataset.drop("y", axis = 1)
    y_test = test_dataset["y"]


    # Preprocessor pipeline
    click.echo("Creating preprocessor pipeline")
    model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight ="balanced"))
    ])

    # Final data validation for label drift in training and testing datasets
    categorical_feats, numerical_feats = validation_after_splitting(X_train, y_train, X_test, y_test)

    # Fitting training data
    click.echo("Fitting the model")
    bank_model = model.fit(X_train, y_train)

    with open(os.path.join(save_location, "../models/bank_model.pickle"), 'wb') as f:
        pickle.dump(bank_model, f)

    # Generating artifacts
    click.echo("Generating artifacts")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    create_conf_matrix(y_test, y_pred, save_location)
    
    # Feature Importance (for logistic regression) Analysis
    # This is a bit tricky with pipelines — we extract processed feature names
    click.echo("Performing feature importance analysis")
    ohe = model.named_steps["preprocessor"].named_transformers_["cat"]["onehot"]
    cat_feature_names = ohe.get_feature_names_out(categorical_feats)
    feature_names = np.concatenate([numerical_feats, cat_feature_names])

    # Get coefficients
    coeffs = model.named_steps["classifier"].coef_[0]

    # Generate feature importance bar chart
    create_feat_imp(feature_names, coeffs, save_location)

    click.echo("Analysis complete!")

if __name__ == '__main__':
    main()

# python scripts/fit_and_predict.py --save_location results/figures/ --preprocessor_pickle results/models/bank_preprocessor.pickle --train_dataset_path data/processed/bank_train.csv --test_dataset_path data/processed/bank_test.csv
