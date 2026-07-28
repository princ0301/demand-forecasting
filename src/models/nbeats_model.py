import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.features.encoding import build_category_maps, encode_categoricals
from src.models.base_model import BaseModel
from src.models.sequence_dataset import TimeSeriesWindowDataset
from src.models.sequence_features import (
    NUMERIC_FEATURE_ORDER,
    build_last_window,
    build_series_arrays,
    compute_price_stats,
    transform_numeric,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

CATEGORICAL_COLS = ["item_id", "dept_id", "store_id", "state_id"]
RAW_INPUT_COLS = ["sales", "sell_price", "wday", "month", "day_of_week", "is_event", "is_weekend", "is_snap"]
SALES_CHANNEL_IDX = 0


class NBeatsBlock(nn.Module):
    def __init__(self, lookback: int, horizon: int, static_input_dim: int, hidden_size: int, num_fc_layers: int) -> None:
        super().__init__()
        layers = []
        in_size = lookback + static_input_dim
        for _ in range(num_fc_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        self.fc = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden_size, lookback)
        self.forecast_head = nn.Linear(hidden_size, horizon)

    def forward(self, residual: torch.Tensor, static_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([residual, static_embedding], dim=1)
        hidden = self.fc(x)
        return self.backcast_head(hidden), self.forecast_head(hidden)


class NBeatsNet(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        category_cardinalities: list[int],
        embedding_dim: int,
        hidden_size: int,
        num_blocks: int,
        num_fc_layers: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, embedding_dim) for card in category_cardinalities])
        static_input_dim = embedding_dim * len(category_cardinalities)
        self.horizon = horizon

        self.blocks = nn.ModuleList([
            NBeatsBlock(lookback, horizon, static_input_dim, hidden_size, num_fc_layers)
            for _ in range(num_blocks)
        ])

    def forward(self, sales_window: torch.Tensor, static_codes: torch.Tensor) -> torch.Tensor:
        embedded = [emb(static_codes[:, i]) for i, emb in enumerate(self.embeddings)]
        static_embedding = torch.cat(embedded, dim=1)

        residual = sales_window
        total_forecast = torch.zeros(sales_window.shape[0], self.horizon, device=sales_window.device)

        for block in self.blocks:
            backcast, forecast = block(residual, static_embedding)
            residual = residual - backcast
            total_forecast = total_forecast + forecast

        return total_forecast


class NBeatsModel(BaseModel):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        hidden_size: int,
        embedding_dim: int,
        num_blocks: int,
        num_fc_layers: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        stride: int,
        device: str | None = None,
    ) -> None:
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_fc_layers = num_fc_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.stride = stride
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: NBeatsNet | None = None
        self.category_maps: dict[str, dict] = {}
        self.price_mean = 0.0
        self.price_std = 1.0
        self.last_window: dict[str, np.ndarray] = {}

    def _prepare(self, df: pl.DataFrame) -> pl.DataFrame:
        selected = df.select(["id", "date"] + CATEGORICAL_COLS + RAW_INPUT_COLS)
        encoded = encode_categoricals(selected, self.category_maps, CATEGORICAL_COLS)
        return transform_numeric(encoded, self.price_mean, self.price_std)

    def fit(self, train: pl.DataFrame) -> None:
        logger.info(f"training nbeats on device: {self.device}")
        self.category_maps = build_category_maps(train, CATEGORICAL_COLS)
        self.price_mean, self.price_std = compute_price_stats(train)

        prepared = self._prepare(train)
        series_arrays, series_static = build_series_arrays(prepared, NUMERIC_FEATURE_ORDER, CATEGORICAL_COLS)

        dataset = TimeSeriesWindowDataset(series_arrays, series_static, self.lookback, self.horizon, self.stride)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        logger.info(f"built {len(dataset)} training windows")

        cardinalities = [len(self.category_maps[col]) for col in CATEGORICAL_COLS]
        self.model = NBeatsNet(
            lookback=self.lookback,
            horizon=self.horizon,
            category_cardinalities=cardinalities,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_blocks=self.num_blocks,
            num_fc_layers=self.num_fc_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for encoder_window, static_codes, target in loader:
                sales_window = encoder_window[:, :, SALES_CHANNEL_IDX].to(self.device)
                static_codes = static_codes.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                preds = self.model(sales_window, static_codes)
                loss = loss_fn(preds, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.info(f"epoch {epoch + 1}/{self.epochs} loss {total_loss / len(loader):.4f}")

        self.last_window = build_last_window(prepared, NUMERIC_FEATURE_ORDER, self.lookback)

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        self.model.eval()

        static_df = test.select(["id"] + CATEGORICAL_COLS).unique()
        static_encoded = encode_categoricals(static_df, self.category_maps, CATEGORICAL_COLS).to_pandas()
        static_map = {
            row.id: np.array([getattr(row, c) for c in CATEGORICAL_COLS], dtype=np.int64)
            for row in static_encoded.itertuples(index=False)
        }

        ids_sorted = sorted(self.last_window.keys())
        dates_sorted = sorted(test["date"].unique().to_list())

        encoder_batch = np.stack([self.last_window[i] for i in ids_sorted])
        sales_batch = encoder_batch[:, :, SALES_CHANNEL_IDX]
        static_batch = np.stack([static_map[i] for i in ids_sorted])

        with torch.no_grad():
            sales_tensor = torch.from_numpy(sales_batch).float().to(self.device)
            static_tensor = torch.from_numpy(static_batch).long().to(self.device)
            preds_log = self.model(sales_tensor, static_tensor).cpu().numpy()

        preds = np.expm1(preds_log)
        preds = np.clip(preds, a_min=0, a_max=None)

        records = [
            (series_id, date, preds[i, j])
            for i, series_id in enumerate(ids_sorted)
            for j, date in enumerate(dates_sorted)
        ]
        result_pd = pd.DataFrame(records, columns=["id", "date", "prediction"])
        return pl.from_pandas(result_pd)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")

        metadata = {
            "lookback": self.lookback,
            "horizon": self.horizon,
            "hidden_size": self.hidden_size,
            "embedding_dim": self.embedding_dim,
            "num_blocks": self.num_blocks,
            "num_fc_layers": self.num_fc_layers,
            "category_maps": self.category_maps,
            "price_mean": self.price_mean,
            "price_std": self.price_std,
            "last_window": self.last_window,
        }
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load(self, path: Path) -> None:
        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        self.lookback = metadata["lookback"]
        self.horizon = metadata["horizon"]
        self.hidden_size = metadata["hidden_size"]
        self.embedding_dim = metadata["embedding_dim"]
        self.num_blocks = metadata["num_blocks"]
        self.num_fc_layers = metadata["num_fc_layers"]
        self.category_maps = metadata["category_maps"]
        self.price_mean = metadata["price_mean"]
        self.price_std = metadata["price_std"]
        self.last_window = metadata["last_window"]

        cardinalities = [len(self.category_maps[col]) for col in CATEGORICAL_COLS]
        self.model = NBeatsNet(
            lookback=self.lookback,
            horizon=self.horizon,
            category_cardinalities=cardinalities,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_blocks=self.num_blocks,
            num_fc_layers=self.num_fc_layers,
        )
        self.model.load_state_dict(torch.load(path / "model.pt", map_location=self.device))
        self.model.to(self.device)