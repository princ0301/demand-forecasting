from src.evaluation.metrics import compute_scale, mape, rmse, rmsse_from_scale
from src.features.encoding import build_category_maps, encode_categoricals
from src.models.base_model import BaseModel
from src.models.baseline import NaiveModel, SeasonalNaiveModel
from src.models.lightgbm_model import LightGBMModel
from src.models.xgboost_model import XGBoostModel


def test_baseline_models_implement_base_model() -> None:
    assert issubclass(NaiveModel, BaseModel)
    assert issubclass(SeasonalNaiveModel, BaseModel)


def test_tree_models_implement_base_model() -> None:
    assert issubclass(LightGBMModel, BaseModel)
    assert issubclass(XGBoostModel, BaseModel)


def test_metrics_are_importable_and_callable() -> None:
    import numpy as np

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    assert rmse(y_true, y_pred) == 0.0
    assert mape(y_true, y_pred) == 0.0

    scale = compute_scale(np.array([1.0, 2.0, 3.0, 4.0]))
    assert rmsse_from_scale(y_true, y_pred, scale) == 0.0