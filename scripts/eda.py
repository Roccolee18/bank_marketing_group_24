
import click
import os
import pandas as pd
import altair as alt

@click.command()
@click.option('--data', type=str, required=True, help="Path to input CSV dataset")
@click.option('--plot-to', type=str, required=True, help="Directory to save plots")
def main(data, plot_to):
    """
    Perform EDA on the dataset and save summary plots.
    """

    # Load dataset
    df = pd.read_csv(data)

    # ============================
    # PRINT BASIC INFO
    # ============================
    print("\n=== HEAD ===")
    print(df.head())

    print("\n=== INFO ===")
    print(df.info())

    save_path = os.path.join(plot_to, "../tables", "data_info.csv")
    df.describe(include='all').to_csv(save_path)

    print("\n=== VALUE COUNTS (y) ===")
    print(df['y'].value_counts())

    print("\n=== NUNIQUE ===")
    print(df.nunique())

    # Unique categorical values
    print("\n=== education unique ===")
    print(df['education'].unique())

    print("\n=== marital unique ===")
    print(df['marital'].unique())

    print("\n=== job unique ===")
    print(df['job'].unique())

    print("\n=== DESCRIBE ===")
    print(df.describe())

    # Ensure plot directory exists
    os.makedirs(plot_to, exist_ok=True)

    alt.data_transformers.enable('vegafusion')

    # ============================
    # PLOT 1: Bar chart of target y
    # ============================
    plot_y = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("y:N", title="Subscribed"),
            x=alt.X("count()", title="Count"),
            color="y:N"
        )
    )
    plot_y.save(os.path.join(plot_to, "target_counts.png"))

    # ============================
    # DENSITY PLOTS (AGE, BALANCE)
    # ============================
    plot_age = (
        alt.Chart(df)
        .transform_density(
            density="age",
            groupby=["y"],
            as_=["age", "density"]
        )
        .mark_area(opacity=0.4)
        .encode(
            x=alt.X("age:Q", title="Age"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("y:N", title="Subscribed")
        )
        .properties(title="Age Density by Subscription")
    )
    plot_age.save(os.path.join(plot_to, "age_density.png"))

    plot_balance = (
        alt.Chart(df)
        .transform_density(
            density="balance",
            groupby=["y"],
            as_=["balance", "density"]
        )
        .mark_area(opacity=0.4)
        .encode(
            x=alt.X("balance:Q", title="Balance"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("y:N", title="Subscribed")
        )
        .properties(title="Balance Density by Subscription")
    )
    plot_balance.save(os.path.join(plot_to, "balance_density.png"))

    # ============================
    # CORRELATION MATRIX
    # ============================
    numeric_cols = df.select_dtypes(include='number').columns
    corr_df = df[numeric_cols].corr().stack().reset_index()
    corr_df.columns = ['var1', 'var2', 'correlation']

    corr_plot = (
        alt.Chart(corr_df)
        .mark_rect()
        .encode(
            x=alt.X('var1:N', title=""),
            y=alt.Y('var2:N', title=""),
            color=alt.Color('correlation:Q', scale=alt.Scale(scheme='redblue')),
            tooltip=['var1', 'var2', 'correlation']
        )
        .properties(title="Correlation Matrix")
    )
    corr_plot.save(os.path.join(plot_to, "correlation_matrix.png"))

    print("\nAll plots saved to:", plot_to)


if __name__ == '__main__':
    main()

# python scripts/eda.py --data data/processed/bank_train.csv --plot-to results/figures
