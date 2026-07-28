import pickle
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.features.encoding import build_category_maps, encode_categoricals
from src.models.base_model import BaseModel

CATEGORICAL_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
STATIC_NUMERIC_COLS = ["sell_price", "wday", "month", "year", "day_of_week", "is_event", "is_weekend", "is_snap"]


def dynamic_feature_columns(lags: list[int], rolling_windows: list[int]) -> list[str]:
    cols = [f"sales_lag_{lag}" for lag in lags]
    for window in rolling_windows:
        cols.append(f"sales_roll_mean_{window}")
        cols.append(f"sales_roll_std_{window}")
    return cols


class RecursiveTreeModel(BaseModel):
    # shared scaffolding for tree-based models that forecast the 28 day horizon
    # recursively, one day at a time, feeding each day's prediction back in as
    # the lag input for the next. subclasses only need to implement how their
    # specific library trains and predicts on a plain numpy feature matrix.

    def __init__(self, lags: list[int], rolling_windows: list[int]) -> None:
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.max_history = max(lags + rolling_windows)

        self.feature_cols = None
        self.category_maps: dict[str, dict] = {}
        self.history_map: dict[str, np.ndarray] = {}

    @abstractmethod
    def _train(self, matrix: np.ndarray, target: np.ndarray, categorical_idx: list[int]) -> None:
        ...

    @abstractmethod
    def _predict_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def _save_native_model(self, path: Path) -> None:
        ...

    @abstractmethod
    def _load_native_model(self, path: Path) -> None:
        ...

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "lags": self.lags,
            "rolling_windows": self.rolling_windows,
            "max_history": self.max_history,
            "feature_cols": self.feature_cols,
            "category_maps": self.category_maps,
            "history_map": self.history_map,
        }
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        self._save_native_model(path)

    def load(self, path: Path) -> None:
        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        self.lags = metadata["lags"]
        self.rolling_windows = metadata["rolling_windows"]
        self.max_history = metadata["max_history"]
        self.feature_cols = metadata["feature_cols"]
        self.category_maps = metadata["category_maps"]
        self.history_map = metadata["history_map"]
        self._load_native_model(path)

    def _store_history(self, train: pl.DataFrame) -> None:
        history_df = (
            train.select(["id", "sales"])
            .group_by("id", maintain_order=True)
            .tail(self.max_history)
            .to_pandas()
        )
        self.history_map = history_df.groupby("id")["sales"].apply(lambda s: s.to_numpy(dtype=np.float32)).to_dict()

    def fit(self, train: pl.DataFrame) -> None:
        self.category_maps = build_category_maps(train, CATEGORICAL_COLS)
        dynamic_cols = dynamic_feature_columns(self.lags, self.rolling_windows)
        self.feature_cols = CATEGORICAL_COLS + STATIC_NUMERIC_COLS + dynamic_cols

        selected = train.select(["id"] + CATEGORICAL_COLS + STATIC_NUMERIC_COLS + dynamic_cols + ["sales"])
        encoded = encode_categoricals(selected, self.category_maps, CATEGORICAL_COLS)

        matrix = encoded.select(self.feature_cols).to_numpy().astype(np.float32)
        target = encoded["sales"].to_numpy().astype(np.float32)
        del encoded, selected

        categorical_idx = list(range(len(CATEGORICAL_COLS)))
        self._train(matrix, target, categorical_idx)
        del matrix, target

        self._store_history(train)

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        selected = test.select(["id", "date"] + CATEGORICAL_COLS + STATIC_NUMERIC_COLS)
        encoded = encode_categoricals(selected, self.category_maps, CATEGORICAL_COLS)
        test_pd = encoded.to_pandas()

        ids_sorted = sorted(test_pd["id"].unique().tolist())
        dates_sorted = sorted(test_pd["date"].unique().tolist())

        history_matrix = np.stack([self.history_map[i] for i in ids_sorted]).astype(np.float32)
        static_cols = CATEGORICAL_COLS + STATIC_NUMERIC_COLS

        daily_results = []
        for current_date in dates_sorted:
            day_df = test_pd[test_pd["date"] == current_date].set_index("id").loc[ids_sorted].reset_index()

            dynamic_values = []
            for lag in self.lags:
                dynamic_values.append(history_matrix[:, -lag])
            for window in self.rolling_windows:
                window_slice = history_matrix[:, -window:]
                dynamic_values.append(window_slice.mean(axis=1))
                dynamic_values.append(window_slice.std(axis=1))

            dynamic_features = np.column_stack(dynamic_values)
            static_features = day_df[static_cols].to_numpy(dtype=np.float32)
            feature_matrix = np.column_stack([static_features, dynamic_features])

            preds = self._predict_batch(feature_matrix)
            preds = np.clip(preds, a_min=0, a_max=None)

            daily_results.append(pd.DataFrame({"id": ids_sorted, "date": current_date, "prediction": preds}))
            history_matrix = np.concatenate([history_matrix[:, 1:], preds.reshape(-1, 1)], axis=1)

        result = pd.concat(daily_results, ignore_index=True)
        return pl.from_pandas(result)