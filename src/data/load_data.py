import pandas as pd
from pathlib import Path
from src.utils.io import load_config, read_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_raw_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(config["paths"]["raw_dir"])
    raw_files = config["raw_files"]

    sales = read_csv(raw_dir / raw_files["sales"])
    calendar = read_csv(raw_dir / raw_files["calendar"])
    prices = read_csv(raw_dir / raw_files["prices"])

    logger.info(f"loaded sales {sales.shape}, calendar {calendar.shape}, prices {prices.shape}")
    return sales, calendar, prices
 
 
if __name__ == "__main__":
    config = load_config()
    load_raw_data(config)