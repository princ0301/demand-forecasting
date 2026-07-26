from pathlib import Path

import modal

from src.utils.io import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = modal.App("demand-forecasting")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "polars", "pandas", "numpy", "pyarrow", "pyyaml", "lightning", "pytorch-forecasting")
    .add_local_python_source("src")
)

volume = modal.Volume.from_name("demand-forecasting-data", create_if_missing=True)


@app.function(image=image, gpu="A10G", volumes={"/data": volume}, timeout=3600)
def train_lstm_remote(config: dict) -> dict:
    import polars as pl

    from src.evaluation.backtest import evaluate_predictions, train_test_split
    from src.models.lstm_model import LSTMModel

    features_path = f"{config['paths']['processed_dir']}/features.parquet"
    df = pl.read_parquet(features_path)

    horizon = config["evaluation"]["horizon"]
    weight_window = config["evaluation"]["weight_window"]
    train, test = train_test_split(df, horizon)

    lstm_config = config["models"]["lstm"]
    model = LSTMModel(
        lookback=lstm_config["lookback"],
        horizon=horizon,
        hidden_size=lstm_config["hidden_size"],
        embedding_dim=lstm_config["embedding_dim"],
        num_layers=lstm_config["num_layers"],
        epochs=lstm_config["epochs"],
        batch_size=lstm_config["batch_size"],
        learning_rate=lstm_config["learning_rate"],
        stride=lstm_config["stride"],
    )

    model.fit(train)
    predictions = model.predict(test)
    result = evaluate_predictions(train, test, predictions, weight_window)
    logger.info(f"lstm (modal gpu) result: {result}")

    model.save(Path("/data/models/lstm"))
    volume.commit()

    return result


@app.function(image=image, gpu="A10G", volumes={"/data": volume}, timeout=3600)
def train_nbeats_remote(config: dict) -> dict:
    import polars as pl

    from src.evaluation.backtest import evaluate_predictions, train_test_split
    from src.models.nbeats_model import NBeatsModel

    features_path = f"{config['paths']['processed_dir']}/features.parquet"
    df = pl.read_parquet(features_path)

    horizon = config["evaluation"]["horizon"]
    weight_window = config["evaluation"]["weight_window"]
    train, test = train_test_split(df, horizon)

    nbeats_config = config["models"]["nbeats"]
    model = NBeatsModel(
        lookback=nbeats_config["lookback"],
        horizon=horizon,
        hidden_size=nbeats_config["hidden_size"],
        embedding_dim=nbeats_config["embedding_dim"],
        num_blocks=nbeats_config["num_blocks"],
        num_fc_layers=nbeats_config["num_fc_layers"],
        epochs=nbeats_config["epochs"],
        batch_size=nbeats_config["batch_size"],
        learning_rate=nbeats_config["learning_rate"],
        stride=nbeats_config["stride"],
    )

    model.fit(train)
    predictions = model.predict(test)
    result = evaluate_predictions(train, test, predictions, weight_window)
    logger.info(f"nbeats (modal gpu) result: {result}")

    model.save(Path("/data/models/nbeats"))
    volume.commit()

    return result


@app.function(image=image, gpu="A10G", memory=32768, volumes={"/data": volume}, timeout=5400)
def train_tft_remote(config: dict) -> dict:
    import polars as pl

    from src.evaluation.backtest import evaluate_predictions, train_test_split
    from src.models.tft_model import TFTModel

    features_path = f"{config['paths']['processed_dir']}/features.parquet"
    df = pl.read_parquet(features_path)

    horizon = config["evaluation"]["horizon"]
    weight_window = config["evaluation"]["weight_window"]
    train, test = train_test_split(df, horizon)

    tft_config = config["models"]["tft"]
    model = TFTModel(
        history_days=tft_config["history_days"],
        lookback=tft_config["lookback"],
        horizon=horizon,
        hidden_size=tft_config["hidden_size"],
        attention_head_size=tft_config["attention_head_size"],
        dropout=tft_config["dropout"],
        hidden_continuous_size=tft_config["hidden_continuous_size"],
        epochs=tft_config["epochs"],
        batch_size=tft_config["batch_size"],
        learning_rate=tft_config["learning_rate"],
    )

    model.fit(train)
    predictions = model.predict(test)
    result = evaluate_predictions(train, test, predictions, weight_window)
    logger.info(f"tft (modal gpu) result: {result}")

    model.save(Path("/data/models/tft"))
    volume.commit()

    return result


REMOTE_FUNCTIONS = {
    "lstm": train_lstm_remote,
    "nbeats": train_nbeats_remote,
    "tft": train_tft_remote,
}


@app.local_entrypoint()
def main(model: str = "lstm", epochs: int = 40):
    config = load_config()
    config["paths"]["processed_dir"] = "/data/processed"
    config["models"][model]["epochs"] = epochs

    result = REMOTE_FUNCTIONS[model].remote(config)
    print(result)