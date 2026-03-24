from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class VisualizationToolkit:
    """
    Generates basic visualizations and saves them to disk.
    """

    ALLOWED_AGG_FUNCS = {"mean", "sum", "count", "min", "max", "median"}

    def __init__(self, dataframe: pd.DataFrame, output_dir: str = "charts") -> None:
        self.dataframe = dataframe.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_column_exists(self, column: str) -> None:
        if column not in self.dataframe.columns:
            raise ValueError(f"Column '{column}' not found in dataset.")

    def _validate_numeric_column(self, column: str) -> None:
        if not pd.api.types.is_numeric_dtype(self.dataframe[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

    def generate_bar_chart(self, x_col: str, y_col: str, agg_func: str = "mean") -> str:
        self._validate_column_exists(x_col)
        self._validate_column_exists(y_col)

        if agg_func not in self.ALLOWED_AGG_FUNCS:
            raise ValueError(
                f"Unsupported aggregation function: {agg_func}. "
                f"Allowed values are: {sorted(self.ALLOWED_AGG_FUNCS)}"
            )

        if agg_func in {"mean", "sum", "min", "max", "median"}:
            self._validate_numeric_column(y_col)

        df_grouped = self.dataframe.groupby(x_col, dropna=False)[y_col].agg(agg_func)

        plt.figure(figsize=(10, 6))
        df_grouped.plot(kind="bar")
        plt.title(f"{agg_func.capitalize()} {y_col} by {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col if agg_func != "count" else f"{agg_func} of {y_col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        file_path = self.output_dir / f"bar_{x_col}_{y_col}_{agg_func}.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()

        return str(file_path)

    def generate_histogram(self, column: str, bins: int = 10) -> str:
        self._validate_column_exists(column)
        self._validate_numeric_column(column)

        plt.figure(figsize=(8, 5))
        self.dataframe[column].dropna().plot(kind="hist", bins=bins)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()

        file_path = self.output_dir / f"hist_{column}.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()

        return str(file_path)

    def generate_scatter_plot(self, x_col: str, y_col: str) -> str:
        self._validate_column_exists(x_col)
        self._validate_column_exists(y_col)
        self._validate_numeric_column(x_col)
        self._validate_numeric_column(y_col)

        plot_df = self.dataframe[[x_col, y_col]].dropna()

        plt.figure(figsize=(8, 5))
        plt.scatter(plot_df[x_col], plot_df[y_col])
        plt.title(f"{y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()

        file_path = self.output_dir / f"scatter_{x_col}_{y_col}.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()

        return str(file_path)