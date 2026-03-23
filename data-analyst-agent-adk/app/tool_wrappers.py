from tools.dataset_manager import DatasetManager
from tools.schema_inspector import SchemaInspector
from tools.eda_toolkit import EDAToolkit
from tools.visualization_toolkit import VisualizationToolkit
from tools.ml_toolkit import MLToolkit

DATASET_PATH = "data/sample_data.csv"


def _load_dataframe():
    dataset_manager = DatasetManager(DATASET_PATH)
    dataset_manager.load_data()
    return dataset_manager.get_dataframe()


def get_columns() -> dict:
    df = _load_dataframe()
    schema_tool = SchemaInspector(df)
    return {"columns": schema_tool.get_column_names()}


def get_summary_statistics() -> dict:
    df = _load_dataframe()
    eda_tool = EDAToolkit(df)
    return eda_tool.get_numeric_summary()


def get_average_sales_by_region() -> dict:
    df = _load_dataframe()
    eda_tool = EDAToolkit(df)
    return eda_tool.groupby_aggregate("region", "sales", "mean")


def get_correlation_matrix() -> dict:
    df = _load_dataframe()
    eda_tool = EDAToolkit(df)
    return eda_tool.get_correlation_matrix()


def plot_sales_by_region() -> dict:
    df = _load_dataframe()
    viz_tool = VisualizationToolkit(df)
    chart_path = viz_tool.generate_bar_chart("region", "sales")
    return {"chart_path": chart_path}


def run_sales_regression() -> dict:
    df = _load_dataframe()
    ml_tool = MLToolkit(df)
    return ml_tool.run_linear_regression(
        feature_cols=["ad_spend", "website_visits"],
        target_col="sales"
    )


def run_churn_classification() -> dict:
    df = _load_dataframe()
    ml_tool = MLToolkit(df)
    return ml_tool.run_random_forest_classification(
        feature_cols=["ad_spend", "website_visits", "sales"],
        target_col="churn"
    )