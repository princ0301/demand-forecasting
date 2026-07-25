import pandas as pd
from pathlib import Path
from src.utils.io import load_config, read_parquet
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["id", "item_id", "store_id", "date", "sales", "sell_price"]
NON_NULLABLE_COLUMNS = ["id", "item_id", "store_id", "date", "sales"]

def check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

def check_non_nullable_columns(df: pd.DataFrame) -> None:
    for col in NON_NULLABLE_COLUMNS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise ValueError(f"column {col} has {null_count} null values, expected zero")

def check_duplicate_rows(df: pd.DataFrame) -> None:
    duplicates = df.duplicated(subset=["id", "date"]).sum()
    if duplicates > 0:
        raise ValueError(f"found {duplicates} duplicate id-date rows")

def check_equal_series_length(df: pd.DataFrame) -> None:
    lengths = df.groupby("id").size()
    if lengths.nunique() != 1:
        # sell_price can legitimately be null before an item existed at a store,
        # but every series must still have the same number of date rows
        logger.info(f"series length range: {lengths.min()} to {lengths.max()}")
        raise ValueError("series do not all have the same number of rows")

def run_validation(df: pd.DataFrame) -> None:
    check_required_columns(df)
    check_non_nullable_columns(df)
    check_duplicate_rows(df)
    check_equal_series_length(df)
    logger.info("all validation checks passed")

if __name__ == "__main__":
    config = load_config()
    path = Path(config["paths"]["interim_dir"]) / "sales_long.parquet"
    dataset = read_parquet(path)
    run_validation(dataset)