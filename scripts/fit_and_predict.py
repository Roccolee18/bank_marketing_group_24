import os
import click
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from deepchecks.tabular.checks import LabelDrift , TrainTestFeatureDrift, MultivariateDrift
from deepchecks.tabular import Dataset, Suite
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, 
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

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
    with open('../results/models/bank_preprocessor.pickle', 'rb') as f:
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
    click.echo("Performing final data validation (label drift)")
    numerical_feats = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_feats = X_train.select_dtypes(include=["object"]).columns
    total_training_dataset = Dataset(pd.concat([X_train, y_train]), label = "y", cat_features = categorical_feats)
    total_testing_dataset = Dataset(pd.concat([X_test, y_test]), label = "y", cat_features = categorical_feats)

    distribution_check = Suite(
        'Drift Detection Suite',
        LabelDrift().add_condition_drift_score_less_than(0.15),
        MultivariateDrift().add_condition_overall_drift_value_less_than(0.15)
    )

    suite_result = distribution_check.run(total_training_dataset, total_testing_dataset)
    has_drift = False

    for check_result in suite_result.results:
        print(f"\nCheck: {check_result.get_header()}")
    
        if hasattr(check_result, 'exception'):
            print(f"  ❌ Check failed to run: {check_result.exception}")
            continue
    
        if check_result.conditions_results:
            for condition_result in check_result.conditions_results:
                status = "✓" if condition_result.is_pass() else "❌"
                click.echo(f"  {status} {condition_result.name}")
                if not condition_result.is_pass():
                    has_drift = True
                    print(f"     Details: {condition_result.details}")
        else:
            print("  No conditions set for this check")

    # Fitting training data
    click.echo("Fitting the model")
    model.fit(X_train, y_train)

    # Generating artifacts
    click.echo("Generating artifacts")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, 
                annot=True, 
                fmt="d", 
                cmap="Blues", 
                xticklabels=["no", "yes"],
                yticklabels=["no", "yes"]
                )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    save_path_sns = os.path.join(save_location + "confusion_matrix.jpg")
    plt.savefig(save_path_sns)
    click.echo("Confusion matrix image generated")
    
    # Feature Importance (for logistic regression) Analysis
    # This is a bit tricky with pipelines — we extract processed feature names
    click.echo("Performing feature importance analysis")
    ohe = model.named_steps["preprocessor"].named_transformers_["cat"]["onehot"]
    cat_feature_names = ohe.get_feature_names_out(categorical_feats)
    feature_names = np.concatenate([numerical_feats, cat_feature_names])

    # Get coefficients
    coeffs = model.named_steps["classifier"].coef_[0]

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": coeffs
    }).sort_values(by="importance", ascending=False)

    print(feat_imp.head(10))
    feat_imp.head(20).plot(kind="bar", x="feature", y="importance", figsize=(10,5))
    plt.title("Feature Importance (Logistic Regression Coefficients)")
    plt.show()
    save_path_feat_imp = os.path.join(save_location + "feature_importance.jpg")
    plt.savefig(save_path_feat_imp, bbox_inches = "tight", dpi = 300)
    click.echo("Feature importance chart generated")
    click.echo("Analysis complete!")

if __name__ == '__main__':
    main()

# python fit_and_predict.py --save_location ../results/figures/ --preprocessor_pickle ../results/models/bank_preprocessor.pickle --train_dataset_path ../data/processed/bank_train.csv --test_dataset_path ../data/processed/bank_test.csv
