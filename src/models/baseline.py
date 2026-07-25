import numpy as np
import pandas as pd
import polars as pl

from src.models.base_model import BaseModel


class NaiveModel(BaseModel):
    def __init__(self) -> None:
        self.last_values: pl.DataFrame | None = None

    def fit(self, train: pl.DataFrame) -> None:
        # train is expected to already be sorted by id, date, this is guaranteed
        # by build_features.py, re-sorting here would double memory use for no benefit
        self.last_values = (
            train.select(["id", "sales"])
            .group_by("id", maintain_order=True)
            .last()
        )

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        dates = test.select(["id", "date"]).unique()
        predictions = dates.join(self.last_values, on="id").rename({"sales": "prediction"})
        return predictions


class SeasonalNaiveModel(BaseModel):
    def __init__(self, season_length: int = 7) -> None:
        self.season_length = season_length
        self.last_season: pl.DataFrame | None = None

    def fit(self, train: pl.DataFrame) -> None:
        # train is expected to already be sorted by id, date, this is guaranteed
        # by build_features.py, re-sorting here would double memory use for no benefit
        self.last_season = (
            train.select(["id", "sales"])
            .group_by("id", maintain_order=True)
            .tail(self.season_length)
        )

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        test_pd = test.select(["id", "date"]).to_pandas()
        season_map = self.last_season.to_pandas().groupby("id")["sales"].apply(lambda s: s.to_numpy()).to_dict()

        results = []
        for series_id, group in test_pd.groupby("id"):
            season = season_map[series_id]
            reps = int(np.ceil(len(group) / len(season)))
            tiled = np.tile(season, reps)[: len(group)]
            group = group.copy()
            group["prediction"] = tiled
            results.append(group)

        return pl.from_pandas(pd.concat(results, ignore_index=True))