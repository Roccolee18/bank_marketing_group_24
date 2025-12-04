import os
import click
import pandas as pd
from ucimlrepo import fetch_ucirepo

@click.command()
@click.option('--dataset_id', type=int, default=222, help="UCI dataset ID to download.")
@click.option(
    '--output-path',
    type=str,
    required=True,
    help="Directory where the CSV file will be saved."
)
@click.option(
    '--output-name',
    type=str,
    default="bank_marketing.csv",
    help="CSV filename to save."
)
def main(dataset_id, output_path, output_name):
    """
    Download a UCI dataset and save it to the specified folder.
    """

    # Create folder if needed
    os.makedirs(output_path, exist_ok=True)

    # Fetch dataset
    click.echo(f"Fetching dataset ID {dataset_id}...")
    dataset = fetch_ucirepo(id=dataset_id)

    # Convert to DataFrame
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)

    # Build full path
    file_path = os.path.join(output_path, output_name)

    # Save CSV
    df.to_csv(file_path, index=False)

    click.echo(f"Dataset saved to: {file_path}")
    click.echo(f"Shape: {df.shape}")
    click.echo(df.head())


if __name__ == '__main__':
    main()

# python dataLoad.py --dataset_id 222 --output-path ./myfolder --output-name marketing.csv
