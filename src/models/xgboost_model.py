"""XGBoost implementation of the recursive tree-model interface."""

from pathlib import Path

import numpy as np

from src.models.recursive_tree_model import RecursiveTreeModel


class XGBoostModel(RecursiveTreeModel):

    def __init__(
        self,
        params: dict | None = None,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        num_boost_round: int = 100,
    ) -> None:
        super().__init__(lags or [], rolling_windows or [])
        self.params = params or {}
        self.num_boost_round = num_boost_round
        self.model = None

    def _train(self, matrix: np.ndarray, target: np.ndarray, categorical_idx: list[int]) -> None:
        import xgboost as xgb

        dataset = xgb.DMatrix(matrix, label=target, feature_names=self.feature_cols)
        self.model = xgb.train(self.params, dataset, num_boost_round=self.num_boost_round)

    def _predict_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        return self.model.predict(xgb.DMatrix(feature_matrix, feature_names=self.feature_cols))

    def _save_native_model(self, path: Path) -> None:
        self.model.save_model(str(path / "model.json"))

    def _load_native_model(self, path: Path) -> None:
        import xgboost as xgb
        self.model = xgb.Booster()
        self.model.load_model(str(path / "model.json"))
