from pathlib import Path

import polars as pl

from src.models.lightgbm_model import LightGBMModel
from src.utils.io import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    config = load_config()
    features_path = Path(config["paths"]["processed_dir"]) / "features.parquet"
    df = pl.read_parquet(features_path)

    # trained on the full history, no held-out window, since a forward-forecasting
    # model should use every available day of signal rather than reserve some for evaluation
    model = LightGBMModel(
        params=config["models"]["lightgbm"]["params"],
        lags=config["features"]["lags"],
        rolling_windows=config["features"]["rolling_windows"],
        num_boost_round=config["models"]["lightgbm"]["num_boost_round"],
    )
    model.fit(df)

    output_dir = Path(config["paths"]["models_dir"]) / "lightgbm_forecast"
    model.save(output_dir)

    # per-series static context (identity + most recent known price) needed to
    # construct synthetic future rows, since future dates don't exist in the data
    last_known = (
        df.select(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "sell_price", "date"])
        .sort(["id", "date"])
        .group_by("id", maintain_order=True)
        .last()
    )
    last_known.write_parquet(output_dir / "last_known.parquet")

    # a small window of real (date, wday) pairs, used to reconstruct M5's weekday
    # encoding for future dates without guessing its convention
    calendar_reference = (
        df.select(["date", "wday"]).unique().sort("date").tail(14)
    )
    calendar_reference.write_parquet(output_dir / "calendar_reference.parquet")

    logger.info(f"saved forecast model and future-generation context to {output_dir}")