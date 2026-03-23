import pandas as pd


class SchemaInspector:
    """
    Inspects dataset structure and basic schema information.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe

    def get_column_names(self) -> list[str]:
        return list(self.dataframe.columns)

    def get_data_types(self) -> dict:
        return {col: str(dtype) for col, dtype in self.dataframe.dtypes.items()}

    def get_missing_values(self) -> dict:
        return self.dataframe.isnull().sum().to_dict()

    def get_sample_rows(self, n: int = 5) -> list[dict]:
        return self.dataframe.head(n).to_dict(orient="records")

    def inspect(self) -> dict:
        return {
            "column_names": self.get_column_names(),
            "data_types": self.get_data_types(),
            "missing_values": self.get_missing_values(),
            "sample_rows": self.get_sample_rows()
        }