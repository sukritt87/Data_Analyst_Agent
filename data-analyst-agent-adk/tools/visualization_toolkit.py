import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class VisualizationToolkit:
    """
    Generates basic visualizations and saves them to disk.
    """

    def __init__(self, dataframe: pd.DataFrame, output_dir: str = "charts") -> None:
        self.dataframe = dataframe
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_bar_chart(self, x_col: str, y_col: str, agg_func: str = "mean") -> str:
        df_grouped = (
            self.dataframe.groupby(x_col)[y_col]
            .agg(agg_func)
        )

        plt.figure()
        df_grouped.plot(kind="bar")
        plt.title(f"{agg_func.capitalize()} {y_col} by {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)

        file_path = self.output_dir / f"bar_{x_col}_{y_col}.png"
        plt.savefig(file_path)
        plt.close()

        return str(file_path)

    def generate_histogram(self, column: str) -> str:
        plt.figure()
        self.dataframe[column].plot(kind="hist", bins=10)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)

        file_path = self.output_dir / f"hist_{column}.png"
        plt.savefig(file_path)
        plt.close()

        return str(file_path)

    def generate_scatter_plot(self, x_col: str, y_col: str) -> str:
        plt.figure()
        plt.scatter(self.dataframe[x_col], self.dataframe[y_col])
        plt.title(f"{y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)

        file_path = self.output_dir / f"scatter_{x_col}_{y_col}.png"
        plt.savefig(file_path)
        plt.close()

        return str(file_path)