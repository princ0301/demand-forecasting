import pandas as pd
from pathlib import Path
from src.data.load_data import load_raw_data
from src.utils.io import load_config, write_parquet
from src.utils.logger import get_logger

logger = get_logger(__name__)
 
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

def filter_category(sales: pd.DataFrame, category: str) -> pd.DataFrame:
    filtered = sales[sales["cat_id"] == category].reset_index(drop=True)
    logger.info(f"filtered to category={category}, {filtered.shape[0]} series remain")
    return filtered

def melt_to_long(sales: pd.DataFrame) -> pd.DataFrame:
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long_df = sales.melt(id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="sales")
    return long_df

def merge_calendar(long_df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    calendar_cols = ["d", "date", "wm_yr_wk", "wday", "month", "year",
                      "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]
    merged = long_df.merge(calendar[calendar_cols], on="d")
    merged["date"] = pd.to_datetime(merged["date"])
    return merged

def merge_prices(long_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    merged = long_df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    return merged

def drop_closed_days(long_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    month = config["data"]["christmas_month"]
    day = config["data"]["christmas_day"]
    mask = (long_df["date"].dt.month == month) & (long_df["date"].dt.day == day)
    filtered = long_df[~mask].reset_index(drop=True)
    logger.info(f"dropped {mask.sum()} closed-store rows")
    return filtered

def build_dataset(config: dict) -> pd.DataFrame:
    sales, calendar, prices = load_raw_data(config)
 
    sales = filter_category(sales, config["data"]["category_filter"])
    long_df = melt_to_long(sales)
    long_df = merge_calendar(long_df, calendar)
    long_df = merge_prices(long_df, prices)
 
    if config["data"]["drop_christmas"]:
        long_df = drop_closed_days(long_df, config)
 
    long_df = long_df.sort_values(["id", "date"]).reset_index(drop=True)
    logger.info(f"final long dataset shape: {long_df.shape}")
    return long_df

if __name__ == "__main__":
    config = load_config()
    dataset = build_dataset(config)
 
    output_path = Path(config["paths"]["interim_dir"]) / "sales_long.parquet"
    write_parquet(dataset, output_path)
    logger.info(f"saved to {output_path}")