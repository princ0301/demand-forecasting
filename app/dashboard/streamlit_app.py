import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import polars as pl
import requests
import streamlit as st

from src.utils.io import load_config

st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")

API_URL = "http://localhost:8000"


@st.cache_data
def load_serving_data() -> pd.DataFrame:
    config = load_config()
    path = Path(config["paths"]["models_dir"]) / "lightgbm_production" / "serving_test.parquet"
    return pl.read_parquet(path).to_pandas()


def get_forecast(item_id: str, store_id: str) -> pd.DataFrame:
    response = requests.get(f"{API_URL}/forecast", params={"item_id": item_id, "store_id": store_id})
    response.raise_for_status()
    forecast_df = pd.DataFrame(response.json()["forecast"])
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    return forecast_df


def get_future_forecast(item_id: str, store_id: str) -> tuple[pd.DataFrame, str]:
    response = requests.get(f"{API_URL}/forecast/future", params={"item_id": item_id, "store_id": store_id})
    response.raise_for_status()
    payload = response.json()
    forecast_df = pd.DataFrame(payload["forecast"])
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    return forecast_df, payload["note"]


serving_data = load_serving_data()

st.title("Demand Forecasting Dashboard")

forecast_type = st.radio(
    "Forecast type",
    ["Historical (backtest vs actual)", "Future (forward forecast)"],
    horizontal=True,
)

col1, col2 = st.columns(2)
with col1:
    item_id = st.selectbox("Item", sorted(serving_data["item_id"].unique()))
with col2:
    available_stores = sorted(serving_data.loc[serving_data["item_id"] == item_id, "store_id"].unique())
    store_id = st.selectbox("Store", available_stores)

if st.button("Get forecast", type="primary"):
    if forecast_type == "Historical (backtest vs actual)":
        try:
            forecast_df = get_forecast(item_id, store_id)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the forecast API: {e}")
        else:
            series_id = f"{item_id}_{store_id}_validation"
            actual = serving_data.loc[serving_data["id"] == series_id, ["date", "sales"]].copy()
            actual["date"] = pd.to_datetime(actual["date"])
            actual = actual.rename(columns={"sales": "actual"})

            merged = actual.merge(forecast_df, on="date", how="inner")
            merged = merged.rename(columns={"predicted_sales": "predicted"})

            errors = (merged["actual"] - merged["predicted"]).abs()
            nonzero = merged["actual"] != 0
            mape = (errors[nonzero] / merged.loc[nonzero, "actual"]).mean() * 100 if nonzero.any() else None

            m1, m2, m3 = st.columns(3)
            m1.metric("Total predicted (28 days)", f"{merged['predicted'].sum():.0f} units")
            m2.metric("Total actual (28 days)", f"{merged['actual'].sum():.0f} units")
            m3.metric("MAPE for this series", f"{mape:.1f}%" if mape is not None else "n/a")

            st.line_chart(merged.set_index("date")[["actual", "predicted"]])
            st.dataframe(merged[["date", "actual", "predicted"]], use_container_width=True, hide_index=True)

    else:
        try:
            forecast_df, note = get_future_forecast(item_id, store_id)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the forecast API: {e}")
        else:
            st.info(note)

            m1, m2 = st.columns(2)
            m1.metric("Total predicted (next 28 days)", f"{forecast_df['predicted_sales'].sum():.0f} units")
            m2.metric("Forecast window", f"{forecast_df['date'].min().date()} to {forecast_df['date'].max().date()}")

            st.line_chart(forecast_df.set_index("date")[["predicted_sales"]])
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)