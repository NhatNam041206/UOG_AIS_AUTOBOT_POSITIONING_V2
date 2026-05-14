from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


class BaseEstimator(ABC):
    short_name = "BASE"
    param_grid: dict[str, list[object]] = {}

    def __init__(self) -> None:
        self.model = None
        self.best_params: dict[str, object] = {}

    @abstractmethod
    def build_model(self, **kwargs):
        raise NotImplementedError

    def train(self, X: list[list[float]], y: list[float], weights: list[float], use_grid_search: bool = True) -> "BaseEstimator":
        if not X:
            raise ValueError("Training requires at least one row of data")
        if use_grid_search and self.param_grid and len(X) >= 6:
            search = GridSearchCV(
                estimator=self.build_model(),
                param_grid=self.param_grid,
                scoring="neg_mean_absolute_error",
                cv=min(3, len(X)),
                n_jobs=1,
            )
            search.fit(X, y, sample_weight=weights)
            self.model = search.best_estimator_
            self.best_params = dict(search.best_params_)
            return self
        model = self.build_model(**self.best_params)
        model.fit(X, y, sample_weight=weights)
        self.model = model
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        if self.model is None:
            raise RuntimeError(f"{self.__class__.__name__} has not been trained")
        return list(self.model.predict(X))


class LinearEstimator(BaseEstimator):
    short_name = "LR"
    param_grid = {"fit_intercept": [True, False]}

    def build_model(self, **kwargs):
        return LinearRegression(**kwargs)


class RandomForestEstimator(BaseEstimator):
    short_name = "RF"
    param_grid = {"n_estimators": [20, 40], "max_depth": [4, 6]}

    def build_model(self, **kwargs):
        return RandomForestRegressor(random_state=42, n_jobs=1, **kwargs)


class XGBoostEstimator(BaseEstimator):
    short_name = "XGB"
    param_grid = {"max_depth": [4, 6], "learning_rate": [0.05, 0.1], "n_estimators": [20]}

    def build_model(self, **kwargs):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - protected by dependency checks
            raise RuntimeError("xgboost must be installed to use XGBoostEstimator") from exc
        return XGBRegressor(
            objective="reg:absoluteerror",
            random_state=42,
            verbosity=0,
            n_jobs=1,
            tree_method="hist",
            **kwargs,
        )
