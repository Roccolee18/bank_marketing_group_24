import click
import os
import pandas as pd
import altair as alt


# def run_eda(input_csv: str, output_dir: str):
#     """
#     Core EDA logic extracted from the Click command.
#     This function can be directly tested with pytest.
#     """


#     # --- Target counts ---
#     plot_y = (
#         alt.Chart(df)
#         .mark_bar()
#         .encode(
#             y=alt.Y("y:N"),
#             x=alt.X("count()"),
#             color="y:N"
#         )
#     )
#     plot_y_path = os.path.join(output_dir, "target_counts.png")
#     plot_y.save(plot_y_path)

#     # --- Age density ---
#     plot_age = (
#         alt.Chart(df)
#         .transform_density("age", groupby=["y"], as_=["age", "density"])
#         .mark_area(opacity=0.4)
#         .encode(
#             x="age:Q",
#             y="density:Q",
#             color="y:N"
#         )
#     )
#     plot_age_path = os.path.join(output_dir, "age_density.png")
#     plot_age.save(plot_age_path)

#     # --- Balance density ---
#     plot_balance = (
#         alt.Chart(df)
#         .transform_density("balance", groupby=["y"], as_=["balance", "density"])
#         .mark_area(opacity=0.4)
#         .encode(
#             x="balance:Q",
#             y="density:Q",
#             color="y:N"
#         )
#     )
#     plot_balance_path = os.path.join(output_dir, "balance_density.png")
#     plot_balance.save(plot_balance_path)

#     # --- Correlation matrix ---
#     numeric_cols = df.select_dtypes(include='number').columns
#     corr_df = df[numeric_cols].corr().stack().reset_index()
#     corr_df.columns = ['var1', 'var2', 'correlation']

#     corr_plot = (
#         alt.Chart(corr_df)
#         .mark_rect()
#         .encode(
#             x='var1:N',
#             y='var2:N',
#             color='correlation:Q',
#         )
#     )
#     corr_plot_path = os.path.join(output_dir, "correlation_matrix.png")
#     corr_plot.save(corr_plot_path)

#     return {
#         "data_info": data_info_path,
#         "plots": [
#             plot_y_path,
#             plot_age_path,
#             plot_balance_path,
#             corr_plot_path,
#         ],
#     }


def bar_chart(df, x_title, y_title, plot_to):
    plot_y = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("y:N", title=y_title),
            x=alt.X("count()", title=x_title),
            color="y:N"
        )
    )
    plot_y.save(plot_to)
    return plot_y


'''
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
'''



def density_plot(df, density_param, group_by, x_title, y_title, plot_title, 
                 color_title, plot_to):
    plot_age = (
        alt.Chart(df)
        .transform_density(
            density=density_param,
            groupby=[group_by],
            as_=[density_param, "density"]
        )
        .mark_area(opacity=0.4)
        .encode(
            x=alt.X(f"{density_param}:Q", title=x_title),
            y=alt.Y("density:Q", title=y_title),
            color=alt.Color("y:N", title=color_title)
        )
        .properties(title=plot_title)
    )
    plot_age.save(plot_to)
    return plot_age


def correlation_plot(df, plot_to):
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
    corr_plot.save(plot_to)
    return corr_plot


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

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create figures folder inside the project root
    table_dir = os.path.join(PROJECT_ROOT, "results/tables")
    os.makedirs(table_dir, exist_ok=True)
    df.describe(include='number').to_csv(os.path.join(table_dir, "data_info.csv"))

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

    # Create figures folder inside the project root
    figures_dir = os.path.join(PROJECT_ROOT, plot_to)
    os.makedirs(figures_dir, exist_ok=True)

    alt.data_transformers.enable('vegafusion')

    bar_chart(df, "Count", "Subscribed", 
              os.path.join(figures_dir, "target_counts.png"))

    density_plot(df, "age", "y", "Age", "Density", 
                 "Age Density by Subscription", "Subscribed", 
                 os.path.join(figures_dir, "age_density.png"))
    density_plot(df, "balance", "y", "Balance", "Density", 
                 "Balance Density by Subscription", "Subscribed", 
                 os.path.join(figures_dir, "balance_density.png"))

    correlation_plot(df, os.path.join(figures_dir, "correlation_matrix.png"))

    print("\nAll plots saved to:", figures_dir)


if __name__ == '__main__':
    main()

# python scripts/eda.py --data data/processed/bank_train.csv --plot-to results/figures
