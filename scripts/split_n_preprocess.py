import click
import os
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


@click.command()
@click.option('--raw-data', type=str, required=True,
              help="Path to the raw bank marketing CSV file.")
@click.option('--data-to', type=str, required=True,
              help="Directory where processed data will be written.")
@click.option('--preprocessor-to', type=str, required=True,
              help="Directory where the preprocessor object will be saved.")
@click.option('--seed', type=int, default=42,
              help="Random seed for reproducibility.")
def main(raw_data, data_to, preprocessor_to, seed):
    """
    Split the Bank Marketing dataset into train and test sets,
    build a preprocessing pipeline, and save transformed data + preprocessor.
    """
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Ensure directories exist
    os.makedirs(data_to, exist_ok=True)
    os.makedirs(preprocessor_to, exist_ok=True)

    # --------------------------------------------------
    # Load raw data and clean it
    # --------------------------------------------------
    click.echo(f"Loading data from {raw_data}...")
    df = pd.read_csv(raw_data)

    df = df.dropna()
    df = df[df['education'] != 'unknown']
    df = df[df['job'] != 'unknown']
    df = df[df['marital'] != 'unknown']

    df["y"] = df["y"].map({"yes": 1, "no": 0})
    df["housing"] = df["housing"].map({"yes": 1, "no": 0})
    df["loan"] = df["loan"].map({"yes": 1, "no": 0})

    # --------------------------------------------------
    # Train-test split
    # --------------------------------------------------
    click.echo("Splitting into train/test...")
    train_df, test_df = train_test_split(
        df,
        train_size=0.70,
        stratify=df["y"],
        random_state=seed
    )

    train_df.to_csv(os.path.join(data_to, "bank_train.csv"), index=False)
    test_df.to_csv(os.path.join(data_to, "bank_test.csv"), index=False)

    # --------------------------------------------------
    # Preprocessing: numeric + categorical
    # --------------------------------------------------

    X_train = train_df.drop(columns=["y"])

    click.echo("Building preprocessing pipeline...")

    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X_train.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(
        steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(
        steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    bank_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )


    pickle.dump(bank_preprocessor, open(os.path.join(preprocessor_to, "bank_preprocessor.pickle"), "wb"))


if __name__ == '__main__':
    main()

# python scripts/split_n_preprocess.py --raw-data data/raw/bank_marketing.csv --data-to data/processed --preprocessor-to results/models