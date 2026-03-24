import glob
import io
import os
from typing import List

import pandas as pd
from google.adk.tools.tool_context import ToolContext

from tools.dataset_manager import DatasetManager
from tools.schema_inspector import SchemaInspector
from tools.eda_toolkit import EDAToolkit
from tools.visualization_toolkit import VisualizationToolkit
from tools.ml_toolkit import MLToolkit

FALLBACK_DATASET_PATH = "data/sample_data.csv"


def _read_csv_safely(file_obj_or_path) -> pd.DataFrame:
    """
    Read a CSV using common encoding fallbacks.
    Supports both file paths and file-like objects.
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error = None

    for encoding in encodings_to_try:
        try:
            if isinstance(file_obj_or_path, (str, os.PathLike)):
                return pd.read_csv(file_obj_or_path, encoding=encoding)
            return pd.read_csv(file_obj_or_path, encoding=encoding)
        except UnicodeDecodeError as e:
            last_error = e
            if hasattr(file_obj_or_path, "seek"):
                file_obj_or_path.seek(0)
        except Exception as e:
            last_error = e
            if hasattr(file_obj_or_path, "seek"):
                file_obj_or_path.seek(0)

    raise ValueError(
        "Unable to read the CSV file with supported encodings "
        "(utf-8, utf-8-sig, cp1252, latin-1)."
    ) from last_error


def _try_convert_numeric(series: pd.Series) -> pd.Series:
    """
    Try to convert numeric-looking text into numeric values.
    Handles values like '$51,415,938', '51,415,938', ' -- ', etc.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    cleaned = (
        series.astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
        .replace(
            {
                "nan": None,
                "None": None,
                "": None,
                "--": None,
                "N/A": None,
                "n/a": None,
            }
        )
    )

    return pd.to_numeric(cleaned, errors="ignore")


def _prepare_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Return a copy of df with requested columns converted to numeric where possible.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = _try_convert_numeric(df[col])
    return df


async def _get_uploaded_csv_names(tool_context: ToolContext) -> list[str]:
    """
    Return uploaded CSV artifact names from the current session.
    """
    try:
        artifact_names = await tool_context.list_artifacts()
    except Exception:
        artifact_names = []

    return [name for name in artifact_names if name.lower().endswith(".csv")]


def _get_local_csv_files() -> list[str]:
    """
    Return local CSV files in /data except the fallback sample file.
    """
    local_csv_files = glob.glob("data/*.csv")
    return [
        f for f in local_csv_files
        if os.path.abspath(f) != os.path.abspath(FALLBACK_DATASET_PATH)
    ]


async def _get_active_dataset_name(tool_context: ToolContext) -> str:
    """
    Return the active dataset name by priority:
    1. uploaded CSV artifact
    2. latest local CSV in /data except fallback
    3. fallback dataset
    """
    csv_files = await _get_uploaded_csv_names(tool_context)
    if csv_files:
        return csv_files[-1]

    local_csv_files = _get_local_csv_files()
    if local_csv_files:
        return max(local_csv_files, key=os.path.getctime)

    return FALLBACK_DATASET_PATH


async def _load_dataframe(tool_context: ToolContext) -> pd.DataFrame:
    """
    Load dataset by priority:
    1. Uploaded CSV from ADK artifacts
    2. Latest local CSV in data/ excluding sample_data.csv
    3. Fallback dataset
    """
    csv_files = await _get_uploaded_csv_names(tool_context)

    if csv_files:
        latest_file = csv_files[-1]
        try:
            artifact = await tool_context.load_artifact(latest_file)
        except Exception:
            artifact = None

        if (
            artifact
            and getattr(artifact, "inline_data", None)
            and getattr(artifact.inline_data, "data", None)
        ):
            return _read_csv_safely(io.BytesIO(artifact.inline_data.data))

    local_csv_files = _get_local_csv_files()
    if local_csv_files:
        latest_file = max(local_csv_files, key=os.path.getctime)
        return _read_csv_safely(latest_file)

    dataset_manager = DatasetManager(FALLBACK_DATASET_PATH)
    dataset_manager.load_data()
    return dataset_manager.get_dataframe()


def _validate_columns_exist(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Validate that all requested columns exist in the dataset.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"These columns were not found in the dataset: {missing}. "
            f"Available columns are: {list(df.columns)}"
        )


def _validate_numeric_columns(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Validate that all requested columns are numeric.
    """
    non_numeric = [
        col for col in columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise ValueError(
            f"These columns must be numeric but are not: {non_numeric}"
        )


async def list_uploaded_artifacts(tool_context: ToolContext) -> dict:
    """
    List uploaded artifacts in the current session and local CSV candidates.
    """
    try:
        names = await tool_context.list_artifacts()
    except Exception:
        names = []

    csv_files = [name for name in names if name.lower().endswith(".csv")]
    local_csv_files = _get_local_csv_files()
    active_dataset = await _get_active_dataset_name(tool_context)

    return {
        "artifacts": names,
        "csv_artifacts": csv_files,
        "local_csv_files": local_csv_files,
        "active_dataset": active_dataset,
    }


async def inspect_dataset(tool_context: ToolContext) -> dict:
    """
    Return a combined schema overview of the active dataset.
    """
    df = await _load_dataframe(tool_context)
    dataset_name = await _get_active_dataset_name(tool_context)
    schema_tool = SchemaInspector(df)

    return {
        "dataset_used": dataset_name,
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": schema_tool.get_column_names(),
        "data_types": schema_tool.get_data_types(),
        "missing_values": schema_tool.get_missing_values(),
        "missing_percentages": schema_tool.get_missing_percentages(),
        "sample_rows": schema_tool.get_sample_rows(),
    }


async def get_columns(tool_context: ToolContext) -> dict:
    df = await _load_dataframe(tool_context)
    dataset_name = await _get_active_dataset_name(tool_context)
    schema_tool = SchemaInspector(df)

    return {
        "dataset_used": dataset_name,
        "columns": schema_tool.get_column_names(),
    }


async def get_data_types(tool_context: ToolContext) -> dict:
    df = await _load_dataframe(tool_context)
    dataset_name = await _get_active_dataset_name(tool_context)
    schema_tool = SchemaInspector(df)

    return {
        "dataset_used": dataset_name,
        "data_types": schema_tool.get_data_types(),
    }


async def get_missing_values(tool_context: ToolContext) -> dict:
    df = await _load_dataframe(tool_context)
    dataset_name = await _get_active_dataset_name(tool_context)
    schema_tool = SchemaInspector(df)

    return {
        "dataset_used": dataset_name,
        "missing_values": schema_tool.get_missing_values(),
        "missing_percentages": schema_tool.get_missing_percentages(),
    }


async def get_summary_statistics(tool_context: ToolContext) -> dict:
    df = await _load_dataframe(tool_context)
    dataset_name = await _get_active_dataset_name(tool_context)

    numeric_candidate_cols = list(df.columns)
    df = _prepare_numeric_columns(df, numeric_candidate_cols)

    eda_tool = EDAToolkit(df)
    return {
        "dataset_used": dataset_name,
        "summary_statistics": eda_tool.get_numeric_summary(),
    }


async def get_categorical_summary(tool_context: ToolContext) -> dict:
    df = await _load_dataframe(tool_context)
    dataset_name = await _get_active_dataset_name(tool_context)
    eda_tool = EDAToolkit(df)

    return {
        "dataset_used": dataset_name,
        "categorical_summary": eda_tool.get_categorical_summary(),
    }


async def get_correlation_matrix(tool_context: ToolContext) -> dict:
    df = await _load_dataframe(tool_context)
    dataset_name = await _get_active_dataset_name(tool_context)

    numeric_candidate_cols = list(df.columns)
    df = _prepare_numeric_columns(df, numeric_candidate_cols)

    eda_tool = EDAToolkit(df)
    return {
        "dataset_used": dataset_name,
        "correlation_matrix": eda_tool.get_correlation_matrix(),
    }


async def groupby_aggregate(
    group_col: str,
    value_col: str,
    agg_func: str,
    tool_context: ToolContext
) -> dict:
    df = await _load_dataframe(tool_context)
    dataset_name = await _get_active_dataset_name(tool_context)

    _validate_columns_exist(df, [group_col, value_col])

    if agg_func in {"mean", "sum", "min", "max", "median"}:
        df = _prepare_numeric_columns(df, [value_col])
        _validate_numeric_columns(df, [value_col])

    eda_tool = EDAToolkit(df)
    result = eda_tool.groupby_aggregate(group_col, value_col, agg_func)

    return {
        "dataset_used": dataset_name,
        "group_col": group_col,
        "value_col": value_col,
        "agg_func": agg_func,
        "result": result,
    }


async def generate_bar_chart(
    x_col: str,
    y_col: str,
    agg_func: str,
    tool_context: ToolContext
) -> dict:
    df = await _load_dataframe(tool_context)
    dataset_name = await _get_active_dataset_name(tool_context)

    _validate_columns_exist(df, [x_col, y_col])

    if agg_func in {"mean", "sum", "min", "max", "median"}:
        df = _prepare_numeric_columns(df, [y_col])
        _validate_numeric_columns(df, [y_col])

    viz_tool = VisualizationToolkit(df)
    chart_path = viz_tool.generate_bar_chart(x_col, y_col, agg_func)

    return {
        "dataset_used": dataset_name,
        "chart_path": chart_path,
        "x_col": x_col,
        "y_col": y_col,
        "agg_func": agg_func,
    }


async def generate_histogram(column: str, tool_context: ToolContext) -> dict:
    df = await _load_dataframe(tool_context)
    dataset_name = await _get_active_dataset_name(tool_context)

    _validate_columns_exist(df, [column])
    df = _prepare_numeric_columns(df, [column])
    _validate_numeric_columns(df, [column])

    viz_tool = VisualizationToolkit(df)
    chart_path = viz_tool.generate_histogram(column)

    return {
        "dataset_used": dataset_name,
        "chart_path": chart_path,
        "column": column,
    }


async def generate_scatter_plot(
    x_col: str,
    y_col: str,
    tool_context: ToolContext
) -> dict:
    df = await _load_dataframe(tool_context)
    dataset_name = await _get_active_dataset_name(tool_context)

    _validate_columns_exist(df, [x_col, y_col])
    df = _prepare_numeric_columns(df, [x_col, y_col])
    _validate_numeric_columns(df, [x_col, y_col])

    viz_tool = VisualizationToolkit(df)
    chart_path = viz_tool.generate_scatter_plot(x_col, y_col)

    return {
        "dataset_used": dataset_name,
        "chart_path": chart_path,
        "x_col": x_col,
        "y_col": y_col,
    }


async def run_linear_regression(
    feature_cols: list[str],
    target_col: str,
    tool_context: ToolContext
) -> dict:
    df = await _load_dataframe(tool_context)
    dataset_name = await _get_active_dataset_name(tool_context)

    _validate_columns_exist(df, feature_cols + [target_col])
    df = _prepare_numeric_columns(df, [target_col])
    _validate_numeric_columns(df, [target_col])

    ml_tool = MLToolkit(df)
    result = ml_tool.run_linear_regression(feature_cols, target_col)

    if isinstance(result, dict):
        result["dataset_used"] = dataset_name
        result["feature_cols"] = feature_cols
        result["target_col"] = target_col
        return result

    return {
        "dataset_used": dataset_name,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "result": result,
    }


async def run_random_forest_classification(
    feature_cols: list[str],
    target_col: str,
    tool_context: ToolContext
) -> dict:
    df = await _load_dataframe(tool_context)
    dataset_name = await _get_active_dataset_name(tool_context)

    _validate_columns_exist(df, feature_cols + [target_col])

    ml_tool = MLToolkit(df)
    result = ml_tool.run_random_forest_classification(feature_cols, target_col)

    if isinstance(result, dict):
        result["dataset_used"] = dataset_name
        result["feature_cols"] = feature_cols
        result["target_col"] = target_col
        return result

    return {
        "dataset_used": dataset_name,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "result": result,
    }