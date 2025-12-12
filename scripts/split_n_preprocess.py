import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --------------------------------------------------
# Functionalized components
# --------------------------------------------------

def load_and_clean(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    df = df.dropna()
    df = df[df['education'] != 'unknown']
    df = df[df['job'] != 'unknown']
    df = df[df['marital'] != 'unknown']

    df["y"] = df["y"].map({"yes": 1, "no": 0})
    df["housing"] = df["housing"].map({"yes": 1, "no": 0})
    df["loan"] = df["loan"].map({"yes": 1, "no": 0})

    return df


def split_data(df: pd.DataFrame, seed: int):
    return train_test_split(
        df,
        train_size=0.70,
        stratify=df["y"],
        random_state=seed
    )


def build_preprocessor(df: pd.DataFrame):
    X = df.drop(columns=["y"])

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )


# --------------------------------------------------
# CLI wrapper
# --------------------------------------------------
import click
from sklearn import set_config

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

    np.random.seed(seed)
    set_config(transform_output="pandas")

    os.makedirs(data_to, exist_ok=True)
    os.makedirs(preprocessor_to, exist_ok=True)

    click.echo("Loading and cleaning data...")
    df = load_and_clean(raw_data)
    click.echo("Splitting data...")
    train_df, test_df = split_data(df, seed)

    train_df.to_csv(os.path.join(data_to, "bank_train.csv"), index=False)
    test_df.to_csv(os.path.join(data_to, "bank_test.csv"), index=False)

    click.echo("Building preprocessor data...")
    preprocessor = build_preprocessor(train_df)
    pickle.dump(preprocessor, open(os.path.join(preprocessor_to, "bank_preprocessor.pickle"), "wb"))
    click.echo("Data cleaned, split and preprocessor built")


if __name__ == '__main__':
    main()

# python scripts/split_n_preprocess.py --raw-data data/raw/bank_marketing.csv --data-to data/processed --preprocessor-to results/models