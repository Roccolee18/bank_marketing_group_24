import os
import sys
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.eda import bar_chart, density_plot, correlation_plot

@pytest.fixture
def dummy_csv(tmp_path):
    df = pd.DataFrame({
        "age": [58, 44, 33, 50, 37, 29, 41, 60, 35, 48],
        "job": ["management","technician","entrepreneur","admin","technician","services","management","retired","admin","technician"],
        "marital": ["married","single","married","single","married","single","married","married","single","single"],
        "education": ["tertiary","secondary","secondary","primary","secondary","secondary","tertiary","tertiary","primary","secondary"],
        "default": ["no"]*10,
        "balance": [23456,2345,4534,6543,5674,5643,45678,7654,65908,78907],
        "housing": ["yes"]*10,
        "loan": ["no","no","yes","no","yes","no","no","yes","no","no"],
        "contact": [""]*10,
        "day_of_week": [5]*10,
        "month": ["may"]*10,
        "duration": [261,151,76,120,85,200,95,300,60,180],
        "campaign": [1]*10,
        "pdays": [-1]*10,
        "previous": [0]*10,
        "poutcome": [""]*10,
        "y": ['no','yes', 'no','no','yes','no','yes', 'no', 'no','no']
    })

    csv_path = tmp_path / "dummy.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# Test Target Counts Plot
def test_bar_chart(tmp_path, dummy_csv):
    temp_data = pd.read_csv(dummy_csv)
    
    # Where to save the plot
    plot_path = tmp_path / "bar.png"

    # Call your bar chart function
    chart = bar_chart(
        df=temp_data,
        x_title="Count of Records",
        y_title="Target Class",
        plot_to=str(plot_path)
    )

    assert plot_path.exists(), "Bar chart PNG file was not created."

    assert chart.mark == "bar", "Chart mark type should be 'bar'."

    chart_dict = chart.to_dict()

    assert chart_dict["encoding"]["x"]["title"] == "Count of Records", "X-axis title is incorrect."
    assert chart_dict["encoding"]["y"]["title"] == "Target Class", "Y-axis title is incorrect."

# Test Balance Density Plot
def test_density_plot(tmp_path, dummy_csv):
    temp_data = pd.read_csv(dummy_csv)

    plot_path = tmp_path / "density.png"

    chart = density_plot(
        df=temp_data,
        density_param="age",
        group_by="y",
        x_title="Age",
        y_title="Density",
        plot_title="Age Density",
        color_title="Subscription",
        plot_to=str(plot_path)
    )

    assert plot_path.exists()

    assert chart.mark.type == "area", "Chart mark type should be 'area'."
    chart_dict = chart.to_dict()

    assert chart_dict["encoding"]["x"]["title"] == "Age", "X-axis title is incorrect."
    assert chart_dict["encoding"]["y"]["title"] == "Density", "Y-axis title is incorrect."


# Test Correlation Matrix Plot
def test_correlation_plot(tmp_path, dummy_csv):
    temp_data = pd.read_csv(dummy_csv)
    plot_path = tmp_path / "corr.png"

    chart = correlation_plot(temp_data, str(plot_path))

    assert plot_path.exists()

    assert chart.mark == "rect", "Chart mark type should be 'rect'."
    chart_dict = chart.to_dict()
    
    assert chart_dict.get("title") == "Correlation Matrix", \
    f"Expected plot title 'Correlation Matrix', got {chart_dict.get('title')}"
