import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix


class MLToolkit:
    """
    Provides lightweight machine learning tools for regression and classification.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe.copy()

    def run_linear_regression(self, feature_cols: list[str], target_col: str) -> dict:
        if target_col not in self.dataframe.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        for col in feature_cols:
            if col not in self.dataframe.columns:
                raise ValueError(f"Feature column '{col}' not found in dataset.")

        X = self.dataframe[feature_cols]
        y = self.dataframe[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        return {
            "model_type": "Linear Regression",
            "features": feature_cols,
            "target": target_col,
            "coefficients": {
                feature: float(coef)
                for feature, coef in zip(feature_cols, model.coef_)
            },
            "intercept": float(model.intercept_),
            "r2_score": float(r2_score(y_test, predictions))
        }

    def run_random_forest_classification(self, feature_cols: list[str], target_col: str) -> dict:
        if target_col not in self.dataframe.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        for col in feature_cols:
            if col not in self.dataframe.columns:
                raise ValueError(f"Feature column '{col}' not found in dataset.")

        X = self.dataframe[feature_cols]
        y = self.dataframe[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        return {
            "model_type": "Random Forest Classification",
            "features": feature_cols,
            "target": target_col,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "feature_importance": {
                feature: float(importance)
                for feature, importance in zip(feature_cols, model.feature_importances_)
            },
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist()
        }