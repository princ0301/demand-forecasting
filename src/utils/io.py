import yaml
import pandas as pd
from pathlib import Path

def load_config(config_path: str = "src/config/config.yaml") -> dict:

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

