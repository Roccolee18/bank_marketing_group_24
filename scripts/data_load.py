import os
import click
import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_and_save_dataset(dataset_id: int, output_path: str, output_name: str):
    """
    Core logic for loading a UCI dataset and saving it locally.
    This function can be imported and tested easily.
    """
    # Create folder if needed
    os.makedirs(output_path, exist_ok=True)

    # Fetch dataset
    dataset = fetch_ucirepo(id=dataset_id)

    # Convert to DataFrame
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)

    # Build full path
    file_path = os.path.join(output_path, output_name)

    # Save CSV
    df.to_csv(file_path, index=False)

    return file_path, df  # return values so pytest can inspect


@click.command()
@click.option('--dataset_id', type=int, default=222)
@click.option('--output-path', type=str, required=True)
@click.option('--output-name', type=str, default="bank_marketing.csv")
def main(dataset_id, output_path, output_name):
    """
    CLI wrapper around the core logic.
    """
    click.echo(f"Fetching dataset ID {dataset_id}...")

    file_path, df = load_and_save_dataset(dataset_id, output_path, output_name)

    click.echo(f"Dataset saved to: {file_path}")
    click.echo(f"Shape: {df.shape}")
    click.echo(df.head().to_string())


if __name__ == '__main__':
    main()