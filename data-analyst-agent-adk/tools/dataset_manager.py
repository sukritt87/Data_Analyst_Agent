import pandas as pd
from pathlib import Path


class DatasetManager:
    """
    Handles dataset loading and basic dataset access for the project.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)
        self.dataframe: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.file_path}")

        if self.file_path.suffix.lower() != ".csv":
            raise ValueError("Only CSV files are supported in this MVP.")

        self.dataframe = pd.read_csv(self.file_path)
        return self.dataframe

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return the loaded DataFrame.
        """
        if self.dataframe is None:
            raise ValueError("No dataset loaded. Call load_data() first.")
        return self.dataframe

    def get_dataset_info(self) -> dict:
        """
        Return basic dataset metadata.
        """
        df = self.get_dataframe()
        return {
            "file_name": self.file_path.name,
            "row_count": df.shape[0],
            "column_count": df.shape[1],
            "columns": list(df.columns),
        }