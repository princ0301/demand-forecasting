from abc import ABC, abstractmethod
from pathlib import Path
import polars as pl

class BaseModel(ABC):
    @abstractmethod
    def fit(self, train: pl.DataFrame) -> None:
        ...

    @abstractmethod
    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        # must return a dataframe with columns: id, date, prediction
        ...

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError