import pandas as pd


class SchemaInspector:
    """
    Inspects dataset structure and basic schema information.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe.copy()

    def get_column_names(self) -> list[str]:
        return list(self.dataframe.columns)

    def get_data_types(self) -> dict:
        return {col: str(dtype) for col, dtype in self.dataframe.dtypes.items()}

    def get_missing_values(self) -> dict:
        return {
            col: int(count)
            for col, count in self.dataframe.isnull().sum().items()
        }

    def get_missing_percentages(self) -> dict:
        total_rows = len(self.dataframe)
        if total_rows == 0:
            return {col: 0.0 for col in self.dataframe.columns}

        return {
            col: float((count / total_rows) * 100)
            for col, count in self.dataframe.isnull().sum().items()
        }

    def get_sample_rows(self, n: int = 5) -> list[dict]:
        n = max(1, int(n))
        return self.dataframe.head(n).to_dict(orient="records")

    def inspect(self, sample_size: int = 5) -> dict:
        return {
            "shape": {
                "rows": int(self.dataframe.shape[0]),
                "columns": int(self.dataframe.shape[1]),
            },
            "column_names": self.get_column_names(),
            "data_types": self.get_data_types(),
            "missing_values": self.get_missing_values(),
            "missing_percentages": self.get_missing_percentages(),
            "sample_rows": self.get_sample_rows(sample_size),
        }