# DON'T  REMOVE THIS OR IT CAUSES AN ERROR WITH DEF RETURN TYPE HINTS!!
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# TODO: Add more models later:
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from simtwo.core.backends.protocol import ChannelModelConfig


@dataclass
class ChannelModelSpec:
    """Serializable metadata for a model configuration or trained model bundle."""
    name: str
    mode: str = "new"
    metadata: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


DEFAULT_MODEL_DIRNAME = "generated_models"


def ensure_model_directory(base_dir: str | Path) -> Path:
    """Create the generated model output directory if needed.
    
    Args:
        base_dir (str | Path): Directory under which the generated-model folder should be created.
    
    Returns:
        Path: Path to the generated model dir.
    """
    out_dir = Path(base_dir) / DEFAULT_MODEL_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def config_to_spec(config: ChannelModelConfig) -> ChannelModelSpec:
    """Convert a runtime channel model config into saved metadata.
    
    Args:
        config (ChannelModelConfig): Model config selected in the GUI.
    
    Returns:
        ChannelModelSpec: ChannelModelSpec containing model name, mode, parameters, and training metadata.
    """
    metadata = {
        "epochs": int(config.epochs),
        "learning_rate": float(config.learning_rate),
        "feature_names": list(config.feature_names or []),
        "target_name": config.target_name,
        "source_model_path": config.model_path,
        "model_kind": config.model_kind,
        "model_family": config.model_family,
    }
    return ChannelModelSpec(name=config.model_name, mode=config.mode, metadata=metadata, params=dict(config.model_params or {}))


def save_model_spec(config: ChannelModelConfig, base_dir: str | Path = ".") -> Path:
    """Write model config metadata to a json file.
    
    Args:
        config (ChannelModelConfig): Model config to serialize.
        base_dir (str | Path): Directory under which the generated model folder should be used.
    
    Returns:
        Path: Path to the written json file.
    """
    spec = config_to_spec(config)
    out_dir = ensure_model_directory(base_dir)
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in spec.name).strip("_")
    if not safe_name:
        safe_name = "channel_model"
    out_path = out_dir / f"{safe_name}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(spec), fh, indent=2)
    return out_path


def load_model_spec(path: str | Path) -> dict[str, Any]:
    """Load model configuration metadata from json.
    
    Args:
        path (str | Path): Path to a json model spec file.
    
    Returns:
        Parsed metadata dictionary.
    
    Raises:
        ValueError: If the file does not contain a json object.
    """
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Model file must contain a JSON object.")
    return data

# TODO: Add more later. Is there a way to make this dynamically?
SUPPORTED_MODEL_KINDS: dict[str, str] = {
    "linear_regression": "SKLearn Linear Regression",
    "random_forest": "SKLearn Random Forest",
}


def normalize_model_kind(model_kind: str) -> str:
    """Normalize user facing estimator names to supported internal keys.
    
    Args:
        model_kind (str): Estimator name or alias from the GUI/configuration.
    
    Returns:
        str: Canonical key from SUPPORTED_MODEL_KINDS.
    
    Raises:
        ValueError: If the estimator kind is unsupported.
    """
    key = str(model_kind or "").strip().lower().replace(" ", "_")
    # TODO: Refactor this. I don't like the way I implemented this:
    aliases = {
        "linear": "linear_regression",
        "linreg": "linear_regression",
        "linearregression": "linear_regression",
        "sklearn_linear_regression": "linear_regression",
        "rf": "random_forest",
        "randomforest": "random_forest",
        "random_forest_regressor": "random_forest",
        "sklearn_random_forest": "random_forest",
    }
    key = aliases.get(key, key)
    if key not in SUPPORTED_MODEL_KINDS:
        raise ValueError(
            f"Unsupported model kind '{model_kind}'. Supported kinds: {', '.join(SUPPORTED_MODEL_KINDS)}"
        )
    return key


def create_estimator(model_kind: str, model_params: dict[str, Any] | None = None) -> Any:
    """Create a sklearn estimator for the requested model kind.
    
    Args:
        model_kind (str): Supported model kind or alias.
        model_params: Optional estimator hyperparameters.
    
    Returns:
        Configured sklearn estimator instance.
    """
    params = dict(model_params or {})
    normalized = normalize_model_kind(model_kind)

    if normalized == "linear_regression":
        fit_intercept = bool(params.get("fit_intercept", True))
        return LinearRegression(fit_intercept=fit_intercept)

    if normalized == "random_forest":
        n_estimators = int(params.get("n_estimators", 200))
        max_depth_raw = params.get("max_depth", None)
        max_depth = None
        if max_depth_raw not in (None, "", 0, "0"):
            max_depth = int(max_depth_raw)
        random_state = int(params.get("random_state", 42))
        min_samples_split = int(params.get("min_samples_split", 2))
        min_samples_leaf = int(params.get("min_samples_leaf", 1))
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

    raise ValueError(f"Unsupported model kind: {model_kind}")


def build_training_arrays(observations: list[dict[str, Any]], feature_names: list[str] | None, target_name: str | None) -> tuple[np.ndarray, np.ndarray, int]:
    """Build numeric feature and target arrays from observation dictionaries.
    
    Args:
        observations: Input rows from the loaded dataset.
        feature_names: Model feature columns to use as predictors.
        target_name (str): Target column to predict.
    
    Returns:
        Tuple (x_arr, y_arr, skipped) containing the feature matrix, target vector, and number of skipped non-numeric/incomplete rows.
    
    Raises:
        ValueError: If no features, no target, or no usable numeric rows are found.
    """
    if not feature_names:
        raise ValueError("Choose at least one feature column before training.")
    if not target_name:
        raise ValueError("Choose a target column before training.")

    xs: list[list[float]] = []
    ys: list[float] = []
    skipped = 0

    for row in observations:
        row_values: list[float] = []
        bad_row = False
        for feature in feature_names:
            value = row.get(feature)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                bad_row = True
                break
            if not np.isfinite(numeric):
                bad_row = True
                break
            row_values.append(numeric)

        if bad_row:
            skipped += 1
            continue

        target_value = row.get(target_name)
        try:
            target_numeric = float(target_value)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if not np.isfinite(target_numeric):
            skipped += 1
            continue

        xs.append(row_values)
        ys.append(target_numeric)

    if not xs:
        raise ValueError("No usable training rows were found. Make sure the selected features and target are numeric.")

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    return x_arr, y_arr, skipped


def _compute_split_counts(n_rows: int, config: ChannelModelConfig) -> tuple[int,int,int]:
    """Compute train, validation, and test row counts.
    
    Args:
        n_rows (int): Number of usable training rows.
        config (ChannelModelConfig): Model configuration containing split fractions.
    
    Returns:
        Tuple (train_count, validation_count, test_count).
    
    Raises:
        ValueError: If the dataset is too small or split fractions are invalid.
    """
    if n_rows < 3:
        raise ValueError("At least 3 usabl rows are required to create train/val/test splits")
    
    train_fraction = max(0.0, float(config.train_fraction))
    validation_fraction = max(0.0, float(config.validation_fraction))
    test_fraction = max(0.0, float(config.test_fraction))
    total = train_fraction + validation_fraction + test_fraction
    if total <= 0.0:
        raise ValueError("Train/validation/test split fractions must sum to positive val.")

    train_fraction /= total
    validation_fraction /= total
    test_fraction /= total

    train_count = int(round(n_rows * train_fraction))
    validation_count = int(round(n_rows * validation_fraction))
    test_count = n_rows - train_count - validation_count

    if train_count < 1:
        train_count = 1
    if validation_count < 1:
        validation_count = 1
    test_count = n_rows - train_count - validation_count

    if test_count < 1:
        test_count = 1
        if validation_count > 1:
            validation_count -= 1
        elif train_count > 1:
            train_count -= 1

    while train_count + validation_count + test_count > n_rows:
        if validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        elif train_count > 1:
            train_count -= 1
        else:
            break

    while train_count + validation_count + test_count < n_rows:
        train_count += 1

    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("This split leaves at least one dataset empty. Increase dataset or adjust the split to continue.")

    return train_count, validation_count, test_count


def fit_model_bundle(observations: list[dict[str, Any]], config: ChannelModelConfig) -> dict[str, Any]:
    """Train an estimator and package it with metrics and metadata.
    
    Args:
        observations: Input rows from the loaded dataset.
        config (ChannelModelConfig): Model configuration describing features, target, split fractions, estimator kind, and hyperparameters.
    
    Returns:
        Dictionary containing the fitted estimator, metadata, split counts, and train/validation/test metrics.
    """
    feature_names = list(config.feature_names or [])
    target_name = config.target_name
    x_arr, y_arr, skipped_rows = build_training_arrays(observations, feature_names, target_name)

    # Put split here?
    train_count, validation_count, test_count = _compute_split_counts(len(y_arr), config)
    train_end = train_count
    validation_end = train_count + validation_count

    x_train = x_arr[:train_end]
    y_train = y_arr[:train_end]
    x_val = x_arr[train_end:validation_end]
    y_val = y_arr[train_end:validation_end]
    x_test = x_arr[validation_end:]
    y_test = y_arr[validation_end:]

    estimator = create_estimator(config.model_kind, config.model_params)
    estimator.fit(x_train, y_train)

    train_preds = np.asarray(estimator.predict(x_train), dtype=float)
    val_preds = np.asarray(estimator.predict(x_val), dtype=float)
    test_preds = np.asarray(estimator.predict(x_test), dtype=float)

    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_preds)))
    validation_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))

    train_r2 = float(r2_score(y_train, train_preds)) if len(y_train) >= 2 else float("nan")
    validation_r2 = float(r2_score(y_val, val_preds)) if len(y_val) >= 2 else float("nan")
    test_r2 = float(r2_score(y_test, test_preds)) if len(y_test) >= 2 else float("nan")

    bundle = {
        "bundle_type": "simtwo_sklearn_model",
        "bundle_version": 1,
        "model_name": config.model_name,
        "model_family": config.model_family,
        "model_kind": normalize_model_kind(config.model_kind),
        "feature_names": feature_names,
        "target_name": target_name,
        "params": dict(config.model_params or {}),
        "metadata": {
            "model_family": config.model_family,
            "epochs": int(config.epochs),
            "learning_rate": float(config.learning_rate),
            "n_samples": int(x_arr.shape[0]),
            "n_features": int(x_arr.shape[1]),
            "skipped_rows": int(skipped_rows),
            "train_count": int(train_count),
            "validation_count": int(validation_count),
            "test_count": int(test_count),
            "train_fraction": float(config.train_fraction),
            "validation_fraction": float(config.validation_fraction),
            "test_fraction": float(config.test_fraction),
            "split_strategy": "chronological",
            "train_rmse": train_rmse,
            "validation_rmse": validation_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "validation_r2": validation_r2,
            "test_r2": test_r2,
        },
        "estimator": estimator,
    }
    
    return bundle


def save_trained_model_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    """Save a trained model bundle as a joblib file.
    
    Args:
        bundle: Model bundle produced by fit_model_bundle.
        base_dir (str | Path): Directory under which the generated model folder should be used.
    
    Returns:
        Path: Path to the written joblib file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    return out_path


def load_trained_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load a trained model bundle from a joblib file.
    
    Args:
        path (str | Path): Path to a saved joblib model bundle.
    
    Returns:
        Loaded bundle dictionary.
    """
    in_path = Path(path)
    data = joblib.load(in_path)
    if not isinstance(data, dict):
        raise ValueError("Saved model file must contain a model bundle dictionary.")
    if "estimator" not in data:
        raise ValueError("Saved model bundle is missing its estimator.")
    return data
