import pandas as pd


class EDAToolkit:
    """
    Provides basic exploratory data analysis tools.
    """

    ALLOWED_AGG_FUNCS = {"mean", "sum", "count", "min", "max", "median"}

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe.copy()

    def get_numeric_summary(self) -> dict:
        numeric_df = self.dataframe.select_dtypes(include=["number"])

        if numeric_df.empty:
            return {"message": "No numeric columns found in dataset."}

        summary_df = numeric_df.describe().transpose()

        return {
            col: {
                stat: float(value) if pd.notnull(value) else None
                for stat, value in row.items()
            }
            for col, row in summary_df.to_dict(orient="index").items()
        }

    def get_categorical_summary(self, top_n: int = 10) -> dict:
        categorical_df = self.dataframe.select_dtypes(include=["object", "category", "bool"])

        if categorical_df.empty:
            return {"message": "No categorical columns found in dataset."}

        summary = {}
        for col in categorical_df.columns:
            value_counts = categorical_df[col].astype(str).value_counts(dropna=False).head(top_n)
            summary[col] = value_counts.to_dict()

        return summary

    def groupby_aggregate(self, group_col: str, value_col: str, agg_func: str = "mean") -> dict:
        if group_col not in self.dataframe.columns:
            raise ValueError(f"Group column '{group_col}' not found in dataset.")

        if value_col not in self.dataframe.columns:
            raise ValueError(f"Value column '{value_col}' not found in dataset.")

        if agg_func not in self.ALLOWED_AGG_FUNCS:
            raise ValueError(
                f"Unsupported aggregation function: {agg_func}. "
                f"Allowed values are: {sorted(self.ALLOWED_AGG_FUNCS)}"
            )

        if agg_func in {"mean", "sum", "min", "max", "median"} and not pd.api.types.is_numeric_dtype(
            self.dataframe[value_col]
        ):
            raise ValueError(
                f"Column '{value_col}' must be numeric for aggregation '{agg_func}'."
            )

        result = (
            self.dataframe.groupby(group_col, dropna=False)[value_col]
            .agg(agg_func)
            .to_dict()
        )

        return {
            str(key): float(value) if isinstance(value, (int, float)) and pd.notnull(value) else value
            for key, value in result.items()
        }

    def get_missing_percentage(self) -> dict:
        total_rows = len(self.dataframe)
        if total_rows == 0:
            return {col: 0.0 for col in self.dataframe.columns}

        missing_percent = (self.dataframe.isnull().sum() / total_rows * 100).to_dict()

        return {
            col: float(value)
            for col, value in missing_percent.items()
        }

    def get_correlation_matrix(self) -> dict:
        numeric_df = self.dataframe.select_dtypes(include=["number"])

        if numeric_df.shape[1] < 2:
            return {
                "message": "At least two numeric columns are required to compute a correlation matrix."
            }

        correlation_matrix = numeric_df.corr(numeric_only=True)

        return correlation_matrix.to_dict()