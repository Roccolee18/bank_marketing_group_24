import os
import click
import pandas as pd
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

@click.command()
@click.option('--save_location', type=str, required=True, help="Directory to save generated artifacts to")
def main(save_location):
    """
    Fitting the model.
    """
    # Preprocessor pipeline
    click.echo("Creating preprocessor pipeline")
    model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight ="balanced"))
    ])

    # Final data validation for label drift in training and testing datasets
    click.echo("Performing final data validation (label drift)")
    categorical_feats = ["job", "marital", "education", "contact", "month", "poutcome"]
    training_dataset = Dataset(pd.concat([X_train, y_train]), label = "y", cat_features = categorical_feats)
    testing_dataset = Dataset(pd.concat([X_test, y_test]), label = "y", cat_features = categorical_feats)

    distribution_check = Suite(
        'Drift Detection Suite',
        LabelDrift().add_condition_drift_score_less_than(0.15),
        MultivariateDrift().add_condition_overall_drift_value_less_than(0.15)
    )

    suite_result = distribution_check.run(training_dataset, testing_dataset)
    has_drift = False
    for check_result in suite_result.results:
        print(f"\nCheck: {check_result.get_header()}")
        
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
    plt.savefig(save_location + "confusion_matrix.jpg")
    click.echo("Confusion matrix image generated")
    
    # Feature Importance (for logistic regression) Analysis
    # This is a bit tricky with pipelines — we extract processed feature names
    click.echo("Performing feature importance analysis")
    ohe = model.named_steps["preprocessor"].named_transformers_["cat"]["onehot"]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([numerical_cols, cat_feature_names])

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
    plt.savefig(save_location + "feature_importance.jpg")
    click.echo("Feature importance chart generated")
    click.echo("Analysis complete!")

if __name__ == '__main__':
    main()

# python model_fitting.py --save_location ./myfolder
