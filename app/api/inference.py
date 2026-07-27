from datetime import date, datetime
from pathlib import Path

import pandas as pd
import polars as pl

from app.api.future_features import build_wday_map, generate_future_frame
from src.models.lightgbm_model import LightGBMModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_series_id(item_id: str, store_id: str) -> str:
    return f"{item_id}_{store_id}_validation"


def to_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


class ForecastService:
    def __init__(self, config: dict) -> None:
        model_dir = Path(config["paths"]["models_dir"]) / "lightgbm_production"

        self.model = LightGBMModel(
            params=config["models"]["lightgbm"]["params"],
            lags=config["features"]["lags"],
            rolling_windows=config["features"]["rolling_windows"],
            num_boost_round=config["models"]["lightgbm"]["num_boost_round"],
        )
        self.model.load(model_dir)
        self.serving_data = pl.read_parquet(model_dir / "serving_test.parquet")
        logger.info("backtest forecast service ready")

    def predict(self, item_id: str, store_id: str) -> list[dict] | None:
        series_id = build_series_id(item_id, store_id)
        subset = self.serving_data.filter(pl.col("id") == series_id)

        if subset.height == 0:
            return None

        predictions = self.model.predict(subset).sort("date")
        return [
            {"date": to_date(row["date"]), "predicted_sales": round(row["prediction"], 2)}
            for row in predictions.to_dicts()
        ]


class FutureForecastService:
    def __init__(self, config: dict) -> None:
        model_dir = Path(config["paths"]["models_dir"]) / "lightgbm_forecast"

        self.model = LightGBMModel(
            params=config["models"]["lightgbm"]["params"],
            lags=config["features"]["lags"],
            rolling_windows=config["features"]["rolling_windows"],
            num_boost_round=config["models"]["lightgbm"]["num_boost_round"],
        )
        self.model.load(model_dir)

        self.last_known = pl.read_parquet(model_dir / "last_known.parquet").to_pandas()
        calendar_reference = pl.read_parquet(model_dir / "calendar_reference.parquet").to_pandas()
        self.wday_map = build_wday_map(calendar_reference)
        self.horizon = config["evaluation"]["horizon"]
        logger.info("future forecast service ready")

    def predict(self, item_id: str, store_id: str) -> list[dict] | None:
        series_id = build_series_id(item_id, store_id)
        series_row = self.last_known[self.last_known["id"] == series_id]

        if series_row.empty:
            return None

        future_frame = generate_future_frame(series_row, self.wday_map, self.horizon)
        predictions = self.model.predict(future_frame).sort("date")
        return [
            {"date": to_date(row["date"]), "predicted_sales": round(row["prediction"], 2)}
            for row in predictions.to_dicts()
        ]