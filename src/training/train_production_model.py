from pathlib import Path

import polars as pl

from src.evaluation.backtest import train_test_split
from src.models.lightgbm_model import LightGBMModel
from src.utils.io import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    config = load_config()
    features_path = Path(config["paths"]["processed_dir"]) / "features.parquet"
    df = pl.read_parquet(features_path)

    horizon = config["evaluation"]["horizon"]
    train, test = train_test_split(df, horizon)

    model = LightGBMModel(
        params=config["models"]["lightgbm"]["params"],
        lags=config["features"]["lags"],
        rolling_windows=config["features"]["rolling_windows"],
        num_boost_round=config["models"]["lightgbm"]["num_boost_round"],
    )
    model.fit(train)

    output_dir = Path(config["paths"]["models_dir"]) / "lightgbm_production"
    model.save(output_dir)

    # the API serves forecasts against this held-out window, it needs the same
    # calendar/price columns the model was evaluated against, saved alongside
    # the model so the API never has to reload the full feature table
    test.write_parquet(output_dir / "serving_test.parquet")

    logger.info(f"saved production model and serving data to {output_dir}")