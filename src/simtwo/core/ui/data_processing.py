from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures

POSIX_TIME_COL = "posix_time"

# TODO: Add more later? Maybe add feature to manually add timezones? idk
COMMON_TIMEZONES = [
    "UTC",
    "Asia/Tokyo",
    "America/Los_Angeles",
    "America/Denver",
    "America/Chicago",
    "America/New_York",
    "Europe/London",
    "Europe/Paris",
]

# Fixed from ms (was doing multiple ocnversions earlier)
POSIX_UNIT_FACTORS_TO_SECONDS = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}

# This is the data class held by the GUI
@dataclass
class ProcessingDataset:
    """Container for a dataset managed by the data processing UI.
    
    The object keeps the dataframe, source file paths, and optional time column metadata together as one transforms, merges, downsamples, or exports intermediate datasets."""
    name: str
    df: pd.DataFrame
    source_paths: list[str] = field(default_factory=list)
    time_column: str | None = None
    timezone: str = ""
    posix_unit: str = ""
    pending_time_column: str | None = None
    pending_timezone: str = ""
    pending_posix_unit: str = ""
    posix_time_ready: bool = False
    notes: list[str] = field(default_factory=list)

    def copy(self, *, name: str | None = None) -> "ProcessingDataset":
        """Handle copy behavior.
        
        Args:
            name (str): Name assigned to the dataset.
        
        Returns:
            ProcessingDataset: The copied dataset.
        """
        return ProcessingDataset(
            name=name or self.name,
            df=self.df.copy(),
            source_paths=list(self.source_paths),
            time_column=self.time_column,
            timezone=self.timezone,
            posix_unit=self.posix_unit,
            pending_time_column=self.pending_time_column,
            pending_timezone=self.pending_timezone,
            pending_posix_unit=self.pending_posix_unit,
            posix_time_ready=self.posix_time_ready,
            notes=list(self.notes),
        )


def _safe_dataset_label(name: str) -> str:
    """Helper for safe dataset labeling.
    
    Args:
        name (str): Name initially assigned to the dataset.
    
    Returns:
        str: Updated dataset label.
    """
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "dataset"


def candidate_time_columns(df: pd.DataFrame) -> list[str]:
    """Auto-determines time columns based on a set of pre-defined substrings."""
    lowered_hits: list[str] = []
    others: list[str] = []
    for col in df.columns:
        text = str(col).lower()
        if any(tok in text for tok in ("time", "timestamp", "date", "epoch", "posix")):
            lowered_hits.append(str(col))
        else:
            others.append(str(col))
    return lowered_hits + others


def ensure_posix_time(dataset: ProcessingDataset) -> None:
    """Ensures posix time is available or valid."""
    if dataset.posix_time_ready and POSIX_TIME_COL in dataset.df.columns:
        dataset.df = (
            dataset.df
            .dropna(subset=[POSIX_TIME_COL])
            .sort_values(POSIX_TIME_COL)
            .reset_index(drop=True)
        )
        return

    if dataset.time_column is None:
        raise ValueError(f"Select a time feature for '{dataset.name}' before processing.")
    if dataset.time_column not in dataset.df.columns:
        raise ValueError(f"Time column '{dataset.time_column}' was not found in '{dataset.name}'.")

    series = dataset.df[dataset.time_column]
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() >= max(1, len(series) // 2):
        factor = POSIX_UNIT_FACTORS_TO_SECONDS.get(dataset.posix_unit, 1.0)
        posix_s = numeric.astype(float) * factor
    else:
        dt = pd.to_datetime(series, errors="coerce")
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(dataset.timezone, nonexistent="shift_forward", ambiguous="NaT")
        else:
            dt = dt.dt.tz_convert(dataset.timezone)

        posix_s = (dt.astype("int64") // 1_000_000_000).astype(float)

    dataset.df[POSIX_TIME_COL] = posix_s
    dataset.df = dataset.df.dropna(subset=[POSIX_TIME_COL]).sort_values(POSIX_TIME_COL).reset_index(drop=True)
    dataset.posix_time_ready = True

    # Important: prevent repeated conversion from multiplying again later.
    dataset.time_column = POSIX_TIME_COL
    dataset.posix_unit = "s"
    dataset.timezone = "UTC"


def numeric_columns(df: pd.DataFrame, *, exclude: Iterable[str] = ()) -> list[str]:
    """Produces a list of only numeric terms in a dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame to inspect.
        exclude: Iterable of strings used for exclude op.

    Returns:
        A list of column names in str format.
    """
    blocked = set(exclude)
    out: list[str] = []
    for col in df.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(str(col))
    return out


def add_derivative(dataset: ProcessingDataset, column: str, *, output_name: str | None = None) -> str:
    """Add derivative of a given feature to the dataset.
    
    Args:
        dataset (ProcessingDataset): Loaded dataset or processing dataset to operate on.
        column (str): Feature to calculate derivative from.
        output_name (str): Value used for output feature name.
    
    Returns:
        The name of the newly-addeed derivative feature.
    
    Raises:
        ValueError: If the operation cannot be completed with the current inputs or state."""
    ensure_posix_time(dataset)
    if column not in dataset.df.columns:
        raise ValueError(f"Column '{column}' not found.")
    y = pd.to_numeric(dataset.df[column], errors="coerce")
    dy = y.diff()
    out_name = output_name or f"{column}_derivative"
    dataset.df[out_name] = dy
    return out_name


def add_quantified_column(dataset: ProcessingDataset, column: str, *, method: str, window: int, output_name: str | None = None) -> str:
    """Adds a new feature to the dataset that performs a quantification operation on one or more features.
    
    Args:
        dataset (ProcessingDataset): Loaded dataset to operate on.
        column (str): Column name to transform.
        method (str): Quantification method selected by the caller or GUI.
        window (int): Window size of aggregation operation.
        output_name (str): Value used for output name.
    
    Returns:
        str: The output feature/target name.
    
    Raises:
        ValueError: If the operation cannot be completed with the current inputs or state.
    """
    if column not in dataset.df.columns:
        raise ValueError(f"Column '{column}' not found.")
    window = max(1, int(window))
    s = pd.to_numeric(dataset.df[column], errors="coerce")
    if method == "rolling_extremal_difference":
        values = s.rolling(window=window, min_periods=1).max() - s.rolling(window=window, min_periods=1).min()
    elif method == "rolling_anchored_difference":
        #values = s - s.shift(window - 1)
        values = s.rolling(window=window, min_periods=1).apply(lambda a: abs(a - a[0]).max(), raw=True)
    elif method == "rolling_stdev":
        values = s.rolling(window=window, min_periods=1).std()
    elif method == "rolling_variance":
        values = s.rolling(window=window, min_periods=1).var()
    else:
        raise ValueError(f"Unsupported quantification method: {method}")
    short = {
        "rolling_extremal_difference": "rexd",
        "rolling_anchored_difference": "rad",
        "rolling_stdev": "rstdev",
        "rolling_variance": "rvar",
    }[method]
    out_name = output_name or f"{column}_{short}_{window}"
    dataset.df[out_name] = values
    return out_name


def remove_columns(dataset: ProcessingDataset, columns: Iterable[str]) -> None:
    """Removes specified columns from the dataset.
    
    Args:
        dataset (ProcessingDataset): Loaded dataset to operate on.
        column (str): Column name(s) to remove.
    
    Raises:
        ValueError: If the operation cannot be completed with the current inputs or state.
    """
    to_drop = [col for col in columns if col in dataset.df.columns]
    if not to_drop:
        raise ValueError("No selected variables were found in the active dataset.")
    dataset.df = dataset.df.drop(columns=to_drop)
    if dataset.time_column in to_drop:
        dataset.time_column = None


def remove_duplicate_timestamps(dataset: ProcessingDataset) -> int:
    """Removes entries from the dataset with identical timestamps."""
    ensure_posix_time(dataset)
    before = len(dataset.df)
    dataset.df = dataset.df.drop_duplicates(subset=[POSIX_TIME_COL], keep="first").reset_index(drop=True)
    return int(before - len(dataset.df))


def drop_nan_rows(dataset: ProcessingDataset, columns: Iterable[str]) -> int:
    """Drops observations from dataset with NAN values."""
    cols = [col for col in columns if col in dataset.df.columns]
    if not cols:
        raise ValueError("Select one or more variables first.")
    before = len(dataset.df)
    dataset.df = dataset.df.dropna(subset=cols).reset_index(drop=True)
    return int(before - len(dataset.df))


def interpolate_missing(dataset: ProcessingDataset, columns: Iterable[str], *, order: int = 1) -> None:
    """Interpolates missing values for a specified feature/target in a dataset.
    
    Args:
        dataset (ProcessingDataset): Loaded dataset to operate on.
        column (str): Column name to interpolate.
        order (int): Interpolation degree (ex: 1 = linear, 2+ = polynomial).
    
    Raises:
        ValueError: If the operation cannot be completed with the current inputs or state.
    """
    cols = [col for col in columns if col in dataset.df.columns]
    if not cols:
        raise ValueError("Select one or more variables first.")
    for col in cols:
        s = pd.to_numeric(dataset.df[col], errors="coerce")
        if order <= 1:
            dataset.df[col] = s.interpolate(method="linear", limit_direction="both")
        else:
            dataset.df[col] = s.interpolate(method="polynomial", order=int(order), limit_direction="both")


def fill_missing(dataset: ProcessingDataset, columns: Iterable[str], *, direction: str) -> None:
    """Fills in missing values for a dataset for any given columns.
    
    Args:
        dataset (ProcessingDataset): Loaded dataset to operate on.
        column (str): Column name to fill.
        direction (str): Fill direction (forward or backward).
    
    Raises:
        ValueError: If the operation cannot be completed with the current inputs or state.
    """
    cols = [col for col in columns if col in dataset.df.columns]
    if not cols:
        raise ValueError("Select one or more variables first.")
    # TODO: Consider removing backfilling option since it's not useful for time series data
    if direction == "forward":
        dataset.df[cols] = dataset.df[cols].ffill()
    elif direction == "backward":
        dataset.df[cols] = dataset.df[cols].bfill()
    elif direction == "both":
        dataset.df[cols] = dataset.df[cols].ffill().bfill()
    else:
        raise ValueError(f"Unsupported fill direction: {direction}")


def polynomial_expand(dataset: ProcessingDataset, columns: Iterable[str], *, degree: int) -> list[str]:
    """Polynomially expands (with interaction terms) given features in a dataset to a specified degree."""
    cols = [col for col in columns if col in dataset.df.columns]
    if len(cols) == 0:
        raise ValueError("Select one or more numeric variables first.")
    X = dataset.df[cols].apply(pd.to_numeric, errors="coerce")
    transformer = PolynomialFeatures(degree=int(degree), include_bias=False)
    expanded = transformer.fit_transform(X)
    names = transformer.get_feature_names_out(cols)
    out_names: list[str] = []
    for idx, raw_name in enumerate(names):
        if raw_name in cols:
            continue
        clean_name = raw_name.replace(" ", " * ")
        if clean_name in dataset.df.columns:
            clean_name = f"poly_{clean_name}_{degree}"
        dataset.df[clean_name] = expanded[:, idx]
        out_names.append(clean_name)
    return out_names


def create_interaction_term(dataset: ProcessingDataset, columns: Iterable[str], *, output_name: str | None = None) -> str:
    """Creates a new dataset feature that is the product of two or more input features."""
    cols = [col for col in columns if col in dataset.df.columns]
    if len(cols) < 2:
        raise ValueError("Select at least two variables.")
    # Prevents categorical features from being used as byte objects
    # TODO: consider adding addl error checking here to avoid casting errors?
    values = dataset.df[cols].apply(pd.to_numeric, errors="coerce")
    out_name = output_name or ("interaction_" + "_x_".join(cols))
    dataset.df[out_name] = values.prod(axis=1)
    return out_name


def create_average_merge(dataset: ProcessingDataset, columns: Iterable[str], *, output_name: str | None = None) -> str:
    """Creates a new dataset feature consisting of the average values of the specified inpuit features."""
    cols = [col for col in columns if col in dataset.df.columns]
    # TODO: verify whether this function can even be called if fewer than two vars are selected
    if len(cols) < 2:
        raise ValueError("Select at least two variables.")
    values = dataset.df[cols].apply(pd.to_numeric, errors="coerce")
    out_name = output_name or ("merged_avg_" + "_".join(cols))
    dataset.df[out_name] = values.mean(axis=1)
    return out_name


def downsample_dataset(dataset: ProcessingDataset, *, method: str, window: int, reference: ProcessingDataset | None = None) -> ProcessingDataset:
    """Downsamples the dataset using decimation or aggregation.
    
    Args:
        dataset (ProcessingDataset): Loaded dataset or processing dataset to operate on.
        method (str): Processing method selected by the caller or GUI.
        window (int): Window size for downsampling method.
        reference (ProcessingDataset): If using a reference set, timestamp values will be aligned to this set.
    
    Returns:
        ProcessingDataset: The newly created processing dataset.
    
    Raises:
        ValueError: If the operation cannot be completed with the current inputs or state.
    """
    ensure_posix_time(dataset)
    df = dataset.df.sort_values(POSIX_TIME_COL).reset_index(drop=True).copy()
    window = max(1, int(window))

    if method == "decimation":
        out = df.iloc[::window].copy().reset_index(drop=True)
    else:
        groups = np.arange(len(df)) // window
        num_cols = numeric_columns(df, exclude=[])
        agg: dict[str, str] = {POSIX_TIME_COL: "first"}
        for col in df.columns:
            if col == POSIX_TIME_COL:
                continue
            if col in num_cols:
                if method == "avg":
                    agg[col] = "mean"
                elif method == "max":
                    agg[col] = "max"
                elif method == "min":
                    agg[col] = "min"
                elif method == "filtered":
                    df[col] = pd.to_numeric(df[col], errors="coerce").rolling(window=window, min_periods=1).mean()
                    agg[col] = "mean"
                else:
                    raise ValueError(f"Unsupported downsample method: {method}")
            else:
                agg[col] = "first"
        out = df.groupby(groups).agg(agg).reset_index(drop=True)

    new_ds = ProcessingDataset(name=f"{dataset.name}_{method}_{window}", df=out, source_paths=list(dataset.source_paths), time_column=dataset.time_column, timezone=dataset.timezone, posix_unit=dataset.posix_unit, notes=list(dataset.notes))

    if reference is not None:
        ensure_posix_time(reference)
        ref = reference.df[[POSIX_TIME_COL]].dropna().drop_duplicates().sort_values(POSIX_TIME_COL)
        aligned = pd.merge_asof(
            ref,
            new_ds.df.sort_values(POSIX_TIME_COL),
            on=POSIX_TIME_COL,
            direction="nearest",
        )
        new_ds.df = aligned.reset_index(drop=True)
        new_ds.name = f"{new_ds.name}_aligned_to_{reference.name}"

    return new_ds


def merge_datasets_on_posix(datasets: list[ProcessingDataset], *, merged_name: str | None = None) -> ProcessingDataset:
    """Merges two or more datasets by POSIX time feature.
    
    Args:
        dataset (ProcessingDataset): Loaded dataset or processing dataset to operate on.
        merged_name (str): Name of the newly-created dataset.
    
    Returns:
        ProcessingDataset: The newly created processing dataset.
    
    Raises:
        ValueError: If the operation cannot be completed with the current inputs or state.
    """
    if len(datasets) < 2:
        raise ValueError("Select at least two datasets to merge.")
    for ds in datasets:
        ensure_posix_time(ds)
        if ds.df.empty:
            raise ValueError(f"Dataset '{ds.name}' is empty.")

    start = max(float(ds.df[POSIX_TIME_COL].min()) for ds in datasets)
    end = min(float(ds.df[POSIX_TIME_COL].max()) for ds in datasets)
    if start >= end:
        raise ValueError("The selected datasets do not overlap in POSIX time.")

    renamed_frames: list[pd.DataFrame] = []
    source_paths: list[str] = []
    for ds in datasets:
        source_paths.extend(ds.source_paths)
        clipped = ds.df[(ds.df[POSIX_TIME_COL] >= start) & (ds.df[POSIX_TIME_COL] <= end)].copy()
        clipped = clipped.dropna(subset=[POSIX_TIME_COL]).drop_duplicates(subset=[POSIX_TIME_COL]).sort_values(POSIX_TIME_COL)
        label = _safe_dataset_label(ds.name)
        rename_map = {col: f"{label}__{col}" for col in clipped.columns if col != POSIX_TIME_COL}
        renamed_frames.append(clipped.rename(columns=rename_map))

    result = renamed_frames[0]
    for next_df in renamed_frames[1:]:
        result = pd.merge_asof(
            result.sort_values(POSIX_TIME_COL),
            next_df.sort_values(POSIX_TIME_COL),
            on=POSIX_TIME_COL,
            direction="nearest",
        )

    return ProcessingDataset(
        name=merged_name or "merged_dataset",
        df=result.reset_index(drop=True),
        source_paths=source_paths,
        time_column=POSIX_TIME_COL,
        timezone="UTC",
        posix_unit="ms",
        pending_time_column=POSIX_TIME_COL,
        pending_timezone="UTC",
        pending_posix_unit="ms",
        posix_time_ready=True,
        notes=["Merged by nearest overlapping POSIX timestamps after trimming non-overlapping tails."],
    )


def descriptive_stats_text(dataset: ProcessingDataset) -> str:
    """Retrieves the string containinmg descriptive statistics for the current processing dataset."""
    if dataset.df.empty:
        return f"Dataset '{dataset.name}' is empty."

    lines: list[str] = []
    lines.append(f"Dataset: {dataset.name}")
    lines.append(f"Rows: {len(dataset.df):,}")
    lines.append(f"Columns: {len(dataset.df.columns):,}")
    lines.append(f"Time column: {dataset.time_column or '<not selected>'}")
    lines.append(f"Timezone: {dataset.timezone}")
    lines.append(f"POSIX unit: {dataset.posix_unit}")
    lines.append("")

    try:
        null_counts = dataset.df.isna().sum()
        if int(null_counts.sum()) > 0:
            lines.append("Missing value counts by column:")
            lines.append(null_counts[null_counts > 0].to_string())
            lines.append("")
    except Exception:
        pass

    try:
        desc = dataset.df.describe(include="all", datetime_is_numeric=True).transpose()
    except TypeError:
        desc = dataset.df.describe(include="all").transpose()

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 240):
        lines.append("Descriptive statistics:")
        lines.append(desc.to_string())

    return "\n".join(lines)
