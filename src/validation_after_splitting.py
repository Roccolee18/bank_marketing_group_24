from deepchecks.tabular.checks import LabelDrift , TrainTestFeatureDrift, MultivariateDrift
from deepchecks.tabular import Dataset, Suite
import click

def validation_after_splitting(X_train, y_train, X_test, y_test):
    click.echo("Performing final data validation (label drift)")
    numerical_feats = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_feats = X_train.select_dtypes(include=["object"]).columns
    validation_categorical_feats = ["job", "marital", "education", "contact", "month", "poutcome"]
    total_training_dataset = Dataset(X_train, label = y_train, cat_features = validation_categorical_feats)
    total_testing_dataset = Dataset(X_test, label = y_test, cat_features = validation_categorical_feats)

    distribution_check = Suite(
        'Drift Detection Suite',
        LabelDrift().add_condition_drift_score_less_than(0.15),
        MultivariateDrift().add_condition_overall_drift_value_less_than(0.15)
    )

    suite_result = distribution_check.run(total_training_dataset, total_testing_dataset)
    has_drift = False

    print(numerical_feats.shape)

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
    return categorical_feats, numerical_feats  