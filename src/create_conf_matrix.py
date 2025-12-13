import click
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, 
    roc_auc_score, confusion_matrix
)

def create_conf_matrix(y_test, y_pred, save_location):
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    cm_df = pd.DataFrame(
    cm, 
    index=[f"Actual_{cls}" for cls in [0, 1]],   # replace [0,1] with your actual class labels if different
    columns=[f"Predicted_{cls}" for cls in [0, 1]]
    )

    save_path = os.path.join(save_location, "../tables", "classification_results.csv")
    df_report.to_csv(save_path, index=True)
    cm_save_path = os.path.join(save_location, "../tables", "confusion_matrix.csv")
    cm_df.to_csv(cm_save_path, index=True)
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
    save_path_sns = os.path.join(save_location + "confusion_matrix.png")
    plt.savefig(save_path_sns)
    click.echo("Confusion matrix image generated")