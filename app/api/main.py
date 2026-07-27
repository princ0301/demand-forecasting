from fastapi import FastAPI, HTTPException

from app.api.inference import ForecastService, FutureForecastService
from app.api.schemas import ForecastResponse, FutureForecastResponse
from src.utils.io import load_config

config = load_config()
service = ForecastService(config)
future_service = FutureForecastService(config)

app = FastAPI(title="Demand Forecasting API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/forecast", response_model=ForecastResponse)
def forecast(item_id: str, store_id: str) -> ForecastResponse:
    result = service.predict(item_id, store_id)
    if result is None:
        raise HTTPException(status_code=404, detail="no data for this item_id/store_id combination")

    return ForecastResponse(item_id=item_id, store_id=store_id, forecast=result)


@app.get("/forecast/future", response_model=FutureForecastResponse)
def forecast_future(item_id: str, store_id: str) -> FutureForecastResponse:
    result = future_service.predict(item_id, store_id)
    if result is None:
        raise HTTPException(status_code=404, detail="no data for this item_id/store_id combination")

    return FutureForecastResponse(item_id=item_id, store_id=store_id, forecast=result)