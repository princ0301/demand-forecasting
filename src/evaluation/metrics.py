import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_scale(y_train: np.ndarray) -> float:
    diffs = np.diff(y_train)
    return float(np.mean(diffs ** 2))


def rmsse_from_scale(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    if scale == 0 or np.isnan(scale):
        return float("nan")
    error = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(error / scale))


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    return rmsse_from_scale(y_true, y_pred, compute_scale(y_train))


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    total = 0.0
    for series_id, score in scores.items():
        if np.isnan(score):
            continue
        total += weights.get(series_id, 0.0) * score
    return total