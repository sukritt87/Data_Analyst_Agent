from pathlib import Path

import pandas as pd


class DatasetManager:
    """
    Handles dataset loading and basic dataset access.
    Supports CSV files for the current MVP.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)
        self.dataframe: pd.DataFrame | None = None

    def _validate_file(self) -> None:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.file_path}")

        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")

        if self.file_path.suffix.lower() != ".csv":
            raise ValueError("Only CSV files are supported in this MVP.")

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset into a pandas DataFrame.
        """
        self._validate_file()
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
        last_error = None

        for encoding in encodings_to_try:
            try:
                self.dataframe = pd.read_csv(self.file_path, encoding=encoding)
                return self.dataframe
            except UnicodeDecodeError as e:
                last_error = e

        raise ValueError(
            "Unable to read the CSV file with supported encodings "
            "(utf-8, utf-8-sig, cp1252, latin-1)."
        ) from last_error
        return self.dataframe

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return the loaded DataFrame.
        """
        if self.dataframe is None:
            raise ValueError("No dataset loaded. Call load_data() first.")
        return self.dataframe

    def reload_data(self) -> pd.DataFrame:
        """
        Force reload the dataset from disk.
        """
        return self.load_data()

    def get_dataset_info(self) -> dict:
        """
        Return basic dataset metadata.
        """
        df = self.get_dataframe()
        return {
            "file_name": self.file_path.name,
            "file_path": str(self.file_path),
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "columns": list(df.columns),
        }