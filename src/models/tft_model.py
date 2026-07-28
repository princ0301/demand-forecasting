import pickle
from datetime import timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import lightning.pytorch as pl_lightning
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

GROUP_ID = "id"
STATIC_CATEGORICALS = ["item_id", "dept_id", "store_id", "state_id"]
KNOWN_REALS = ["time_idx", "sell_price", "wday", "month", "day_of_week", "is_event", "is_weekend", "is_snap"]
UNKNOWN_REALS = ["sales"]
ALL_COLS = [GROUP_ID, "date"] + STATIC_CATEGORICALS + KNOWN_REALS[1:] + UNKNOWN_REALS


class EpochLoggerCallback(pl_lightning.Callback):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        loss = trainer.callback_metrics.get("train_loss_epoch") or trainer.callback_metrics.get("train_loss")
        loss_value = f"{loss:.4f}" if loss is not None else "n/a"
        logger.info(f"epoch {trainer.current_epoch + 1}/{trainer.max_epochs} loss {loss_value}")


class TFTModel(BaseModel):
    def __init__(
        self,
        history_days: int,
        lookback: int,
        horizon: int,
        hidden_size: int,
        attention_head_size: int,
        dropout: float,
        hidden_continuous_size: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> None:
        self.history_days = history_days
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.reference_date = None
        self.price_mean = 0.0
        self.training_dataset = None
        self.model = None
        self.encoder_history_df: pd.DataFrame | None = None

    def _add_time_idx(self, df: pd.DataFrame) -> pd.DataFrame:
        df["time_idx"] = (df["date"] - self.reference_date).dt.days
        return df

    def fit(self, train: pl.DataFrame) -> None:
        self.reference_date = train["date"].min()

        cutoff = train["date"].max() - timedelta(days=self.history_days)
        train_recent = train.filter(pl.col("date") >= cutoff)

        train_pd = train_recent.select(ALL_COLS).to_pandas()
        for col in STATIC_CATEGORICALS:
            train_pd[col] = train_pd[col].astype(str)
        train_pd = self._add_time_idx(train_pd)
        train_pd["sales"] = train_pd["sales"].astype(float)

        self.price_mean = float(train_pd["sell_price"].mean())
        train_pd["sell_price"] = train_pd["sell_price"].fillna(self.price_mean)

        self.training_dataset = TimeSeriesDataSet(
            train_pd,
            time_idx="time_idx",
            target="sales",
            group_ids=[GROUP_ID],
            min_encoder_length=self.lookback,
            max_encoder_length=self.lookback,
            min_prediction_length=self.horizon,
            max_prediction_length=self.horizon,
            static_categoricals=STATIC_CATEGORICALS,
            time_varying_known_reals=KNOWN_REALS,
            time_varying_unknown_reals=UNKNOWN_REALS,
            target_normalizer=GroupNormalizer(groups=[GROUP_ID], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        train_loader = self.training_dataset.to_dataloader(train=True, batch_size=self.batch_size, num_workers=2)

        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=QuantileLoss(),
            optimizer="adam",
        )

        torch.set_float32_matmul_precision("medium")
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = pl_lightning.Trainer(
            max_epochs=self.epochs,
            accelerator=accelerator,
            devices=1,
            gradient_clip_val=0.1,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
            callbacks=[EpochLoggerCallback()],
        )
        logger.info(f"training tft on accelerator: {accelerator}")
        trainer.fit(self.model, train_dataloaders=train_loader)

        self.encoder_history_df = (
            train_pd.sort_values(["id", "time_idx"])
            .groupby("id", group_keys=False)
            .tail(self.lookback)
            .reset_index(drop=True)
        )

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        test_pd = test.select(ALL_COLS).to_pandas()
        for col in STATIC_CATEGORICALS:
            test_pd[col] = test_pd[col].astype(str)
        test_pd = self._add_time_idx(test_pd)
        test_pd["sell_price"] = test_pd["sell_price"].fillna(self.price_mean)
        test_pd["sales"] = 0.0

        combined = pd.concat([self.encoder_history_df, test_pd], ignore_index=True)
        combined = combined.sort_values(["id", "time_idx"]).reset_index(drop=True)

        predict_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, combined, predict=True, stop_randomization=True,
        )
        predict_loader = predict_dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)

        result = self.model.predict(predict_loader, mode="prediction", return_index=True)
        raw_predictions = result.output
        prediction_index = result.index
        predictions = raw_predictions.cpu().numpy()

        dates_sorted = sorted(test["date"].unique().to_list())
        ids_in_order = prediction_index[GROUP_ID].tolist()

        records = [
            (series_id, date, max(float(predictions[i, j]), 0.0))
            for i, series_id in enumerate(ids_in_order)
            for j, date in enumerate(dates_sorted)
        ]
        result_pd = pd.DataFrame(records, columns=["id", "date", "prediction"])
        return pl.from_pandas(result_pd)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")

        with open(path / "training_dataset.pkl", "wb") as f:
            pickle.dump(self.training_dataset, f)

        metadata = {
            "reference_date": self.reference_date,
            "price_mean": self.price_mean,
            "encoder_history_df": self.encoder_history_df,
            "hidden_size": self.hidden_size,
            "attention_head_size": self.attention_head_size,
            "dropout": self.dropout,
            "hidden_continuous_size": self.hidden_continuous_size,
            "lookback": self.lookback,
            "horizon": self.horizon,
        }
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load(self, path: Path) -> None:
        with open(path / "training_dataset.pkl", "rb") as f:
            self.training_dataset = pickle.load(f)

        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        self.reference_date = metadata["reference_date"]
        self.price_mean = metadata["price_mean"]
        self.encoder_history_df = metadata["encoder_history_df"]
        self.hidden_size = metadata["hidden_size"]
        self.attention_head_size = metadata["attention_head_size"]
        self.dropout = metadata["dropout"]
        self.hidden_continuous_size = metadata["hidden_continuous_size"]
        self.lookback = metadata["lookback"]
        self.horizon = metadata["horizon"]

        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=QuantileLoss(),
            optimizer="adam",
        )
        self.model.load_state_dict(torch.load(path / "model.pt"))