import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class MLToolkit:
    """
    Provides lightweight machine learning tools for regression and classification.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe.copy()

    def _validate_columns_exist(self, feature_cols: list[str], target_col: str) -> None:
        missing = [col for col in feature_cols + [target_col] if col not in self.dataframe.columns]
        if missing:
            raise ValueError(f"These columns were not found in the dataset: {missing}")

    def _prepare_subset(self, feature_cols: list[str], target_col: str) -> pd.DataFrame:
        df = self.dataframe[feature_cols + [target_col]].copy()

        if df.empty:
            raise ValueError("Dataset is empty after selecting the requested columns.")

        if df[target_col].isnull().all():
            raise ValueError(f"Target column '{target_col}' contains only missing values.")

        df = df.dropna(subset=[target_col])

        if len(df) < 5:
            raise ValueError("Not enough rows available after removing missing target values.")

        return df

    def run_linear_regression(self, feature_cols: list[str], target_col: str) -> dict:
        self._validate_columns_exist(feature_cols, target_col)

        df = self._prepare_subset(feature_cols, target_col)

        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError(f"Target column '{target_col}' must be numeric for linear regression.")

        numeric_features = [
            col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])
        ]
        categorical_features = [
            col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])
        ]

        if not feature_cols:
            raise ValueError("At least one feature column is required.")

        X = df[feature_cols]
        y = df[target_col]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                ),
            ],
            remainder="drop",
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        return {
            "model_type": "Linear Regression",
            "features": feature_cols,
            "target": target_col,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "r2_score": float(r2_score(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(mean_squared_error(y_test, predictions, squared=False)),
        }

    def run_random_forest_classification(self, feature_cols: list[str], target_col: str) -> dict:
        self._validate_columns_exist(feature_cols, target_col)

        df = self._prepare_subset(feature_cols, target_col)

        if not feature_cols:
            raise ValueError("At least one feature column is required.")

        target_unique = df[target_col].nunique(dropna=True)
        if target_unique < 2:
            raise ValueError("Target column must contain at least two classes for classification.")

        numeric_features = [
            col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])
        ]
        categorical_features = [
            col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])
        ]

        X = df[feature_cols]
        y = df[target_col].astype(str)

        stratify = y if y.nunique() > 1 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                ),
            ],
            remainder="drop",
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        classifier = model.named_steps["classifier"]
        feature_importance = {}

        try:
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            feature_importance = {
                str(feature): float(importance)
                for feature, importance in zip(feature_names, classifier.feature_importances_)
            }
        except Exception:
            feature_importance = {}

        return {
            "model_type": "Random Forest Classification",
            "features": feature_cols,
            "target": target_col,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "feature_importance": feature_importance,
            "classes": sorted(y.unique().tolist()),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        }