import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Union, List, Dict, Optional, Any


class QuantitativeResearchToolkit:
    """
    A toolkit for quantitative research, focusing on financial data analysis.

    Supports EDA, statistical analysis, machine learning, and results documentation.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the toolkit with a dataset.

        Args:
            data (pd.DataFrame): The dataset to analyze.
        """
        self.data = data
        self.target = None
        self.features = None
        self.model = None
        self.results = {}

    def automate_eda(
        self, target: Optional[str] = None, save_path: Optional[str] = None
    ) -> Dict:
        """
        Perform automated exploratory data analysis (EDA).

        Args:
            target (Optional[str]): The target variable for analysis.
            save_path (Optional[str]): Path to save EDA results (images, JSON, etc.).

        Returns:
            Dict: Summary statistics and insights from EDA.
        """
        eda_results = {}

        # Basic statistics
        eda_results["summary_stats"] = self.data.describe().to_dict()

        # Correlation matrix
        if target:
            correlation_matrix = self.data.corr()
            eda_results["correlation_with_target"] = correlation_matrix[target].to_dict()

            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            if save_path:
                plt.savefig(os.path.join(save_path, "correlation_matrix.png"))
            plt.close()

        # Missing values
        eda_results["missing_values"] = self.data.isnull().sum().to_dict()

        # Distribution plots
        if save_path:
            for column in self.data.columns:
                plt.figure(figsize=(8, 6))
                sns.histplot(self.data[column], kde=True)
                plt.title(f"Distribution of {column}")
                plt.savefig(os.path.join(save_path, f"{column}_distribution.png"))
                plt.close()

        return eda_results

    def train_model(
        self,
        target: str,
        model: RegressorMixin,
        features: Optional[List[str]] = None,
        rolling_window: Optional[int] = None,
        expanding_window: bool = False,
        test_size: float = 0.2,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        save_path: Optional[str] = None,
    ) -> Dict:
        """
        Train a machine learning model on the dataset.

        Args:
            target (str): The target variable.
            model (RegressorMixin): A scikit-learn regression model or any custom model
                                    inheriting from RegressorMixin.
            features (Optional[List[str]]): List of features to use. If None, use all columns except target.
            rolling_window (Optional[int]): Size of the rolling window for time series data.
            expanding_window (bool): Whether to use an expanding window for time series data.
            test_size (float): Proportion of data to use for testing.
            param_grid (Optional[Dict[str, List[Any]]]): Hyperparameter grid for tuning.
            save_path (Optional[str]): Path to save model and results.

        Returns:
            Dict: Model performance metrics and trained model.
        """
        self.target = target
        self.features = features if features else [col for col in self.data.columns if col != target]

        # Prepare data
        X = self.data[self.features]
        y = self.data[target]

        # Split data into training, validation, and test sets
        if rolling_window or expanding_window:
            # Time series split
            tscv = TimeSeriesSplit(n_splits=5)
            metrics = []
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Further split training set into training and validation sets
                val_size = int(len(X_train) * test_size)
                X_train, X_val = X_train[:-val_size], X_train[-val_size:]
                y_train, y_val = y_train[:-val_size], y_train[-val_size:]

                # Scale data
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                X_test = scaler.transform(X_test)

                # Hyperparameter tuning
                if param_grid:
                    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=3)
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    best_model = model

                # Validate on validation set
                y_val_pred = best_model.predict(X_val)
                val_mse = mean_squared_error(y_val, y_val_pred)
                val_r2 = r2_score(y_val, y_val_pred)

                # Test on test set
                y_test_pred = best_model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                metrics.append({
                    "validation_mse": val_mse,
                    "validation_r2": val_r2,
                    "test_mse": test_mse,
                    "test_r2": test_r2,
                    "best_params": grid_search.best_params_ if param_grid else None,
                })
        else:
            # Standard train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

            # Scale data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # Hyperparameter tuning
            if param_grid:
                grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=3)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            # Validate on validation set
            y_val_pred = best_model.predict(X_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            # Test on test set
            y_test_pred = best_model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            metrics = [{
                "validation_mse": val_mse,
                "validation_r2": val_r2,
                "test_mse": test_mse,
                "test_r2": test_r2,
                "best_params": grid_search.best_params_ if param_grid else None,
            }]

        # Save results
        self.results["model_performance"] = metrics
        self.model = best_model

        if save_path:
            self._save_results(save_path)

        return self.results

    def _save_results(self, save_path: str):
        """
        Save results, models, and documentation to the specified path.

        Args:
            save_path (str): Path to save results.
        """
        os.makedirs(save_path, exist_ok=True)

        # Save model
        if self.model:
            with open(os.path.join(save_path, "model.pkl"), "wb") as f:
                pickle.dump(self.model, f)

        # Save results
        with open(os.path.join(save_path, "results.json"), "w") as f:
            json.dump(self.results, f)

        # Save feature importance (if applicable)
        if hasattr(self.model, "feature_importances_"):
            feature_importance = dict(zip(self.features, self.model.feature_importances_))
            with open(os.path.join(save_path, "feature_importance.json"), "w") as f:
                json.dump(feature_importance, f)


# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv("financial_data.csv")

    # Initialize toolkit
    toolkit = QuantitativeResearchToolkit(data)

    # Perform EDA
    eda_results = toolkit.automate_eda(target="target_column", save_path="eda_results")

    # Define model and parameter grid
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
    }

    # Train model
    model_results = toolkit.train_model(
        target="target_column",
        model=model,
        rolling_window=30,
        param_grid=param_grid,
        save_path="model_results",
    )

