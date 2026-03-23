import pandas as pd


class EDAToolkit:
    """
    Provides basic exploratory data analysis tools.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe

    def get_numeric_summary(self) -> dict:
        numeric_df = self.dataframe.select_dtypes(include=["number"])
        return numeric_df.describe().to_dict()

    def get_categorical_summary(self) -> dict:
        categorical_df = self.dataframe.select_dtypes(include=["object", "category"])
        summary = {}

        for col in categorical_df.columns:
            summary[col] = categorical_df[col].value_counts().to_dict()

        return summary

    def groupby_aggregate(self, group_col: str, value_col: str, agg_func: str = "mean") -> dict:
        if group_col not in self.dataframe.columns:
            raise ValueError(f"Group column '{group_col}' not found in dataset.")

        if value_col not in self.dataframe.columns:
            raise ValueError(f"Value column '{value_col}' not found in dataset.")

        if agg_func not in ["mean", "sum", "count", "min", "max"]:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")

        result = (
            self.dataframe.groupby(group_col)[value_col]
            .agg(agg_func)
            .to_dict()
        )

        return result
    
    def get_missing_percentage(self) -> dict:
        total_rows = len(self.dataframe)
        missing_percent = (
            self.dataframe.isnull().sum() / total_rows * 100
        ).to_dict()
        return missing_percent


    def get_correlation_matrix(self) -> dict:
        numeric_df = self.dataframe.select_dtypes(include=["number"])
        correlation_matrix = numeric_df.corr()
        return correlation_matrix.to_dict()