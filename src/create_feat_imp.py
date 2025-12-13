import pandas as pd
import click
import os
import matplotlib.pyplot as plt

def create_feat_imp(feature_names, coeffs, save_location):
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": coeffs
    }).sort_values(by="importance", ascending=False)

    print(feat_imp.head(10))
    feat_imp.head(20).plot(kind="bar", x="feature", y="importance", figsize=(10,5))
    plt.title("Feature Importance (Logistic Regression Coefficients)")
    plt.show()
    save_path_feat_imp = os.path.join(save_location + "feature_importance.png")
    plt.savefig(save_path_feat_imp, bbox_inches = "tight", dpi = 300)
    click.echo("Feature importance chart generated")