from datetime import date

from pydantic import BaseModel


class ForecastPoint(BaseModel):
    date: date
    predicted_sales: float


class ForecastResponse(BaseModel):
    item_id: str
    store_id: str
    forecast: list[ForecastPoint]


class FutureForecastResponse(BaseModel):
    item_id: str
    store_id: str
    forecast: list[ForecastPoint]
    note: str = (
        "Beyond the dataset's historical range, this forecast assumes no promotional "
        "events and a constant price carried forward from the last known value. A real "
        "deployment would replace these with the retailer's actual planning calendar."
    )