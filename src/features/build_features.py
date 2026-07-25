import polars as pl
from pathlib import Path
from src.features.calendar_features import add_calendar_features
from src.features.lag_features import add_lag_features
from src.features.rolling_features import add_rolling_features
from src.utils.io import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def build_feature_table(config: dict) -> pl.DataFrame:
    interim_path = Path(config["paths"]["interim_dir"]) / "sales_long.parquet"
    df = pl.read_parquet(interim_path)
    df = df.sort(["id", "date"])
    logger.info(f"loaded interim dataset: {df.shape}")

    df = add_calendar_features(df)
    df = add_lag_features(df, config["features"]["lags"])
    df = add_rolling_features(df, config["features"]["rolling_windows"])
    logger.info(f"feature table built: {df.shape}")

    return df

if __name__ == "__main__":
    config = load_config()
    features_df = build_feature_table(config)

    output_path = Path(config["paths"]["processed_dir"]) / "features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.write_parquet(output_path)
    logger.info(f"saved to {output_path}")