"""LightGBM implementation of the recursive tree-model interface."""

from pathlib import Path

import numpy as np

from src.models.recursive_tree_model import RecursiveTreeModel


class LightGBMModel(RecursiveTreeModel):

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
        import lightgbm as lgb

        dataset = lgb.Dataset(matrix, label=target, categorical_feature=categorical_idx)
        self.model = lgb.train(self.params, dataset, num_boost_round=self.num_boost_round)

    def _predict_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        return self.model.predict(feature_matrix)

    def _save_native_model(self, path: Path) -> None:
        self.model.save_model(str(path / "booster.txt"))

    def _load_native_model(self, path: Path) -> None:
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=str(path / "booster.txt"))
