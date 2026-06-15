from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

from simtwo.core.backends.protocol import ChannelModelConfig, SimulationBackend
from simtwo.core.modeling.model import SUPPORTED_MODEL_KINDS
from simtwo.core.state import UIState
from simtwo.core.ui.data_processing import (
    COMMON_TIMEZONES,
    POSIX_TIME_COL,
    ProcessingDataset,
    add_derivative,
    add_quantified_column,
    candidate_time_columns,
    create_average_merge,
    create_interaction_term,
    descriptive_stats_text,
    downsample_dataset,
    drop_nan_rows,
    ensure_posix_time,
    fill_missing,
    interpolate_missing,
    merge_datasets_on_posix,
    polynomial_expand,
    remove_columns,
    remove_duplicate_timestamps,
)
from simtwo.core.ui.dialogs import draw_about_popup, draw_csv_headers_window, draw_data_processing_window, draw_import_csv_picker_window, draw_import_format_window
from simtwo.core.ui.panels import draw_left_panel, draw_menu_bar, draw_model_config_panel, draw_right_panel

from simtwo.core.ui.plots import save_poincare_plot, save_timing_plot

import glfw 
import imgui 
import pandas as pd
from imgui.integrations.glfw import GlfwRenderer 
from OpenGL import GL

import filedialpy


@dataclass
class ObserverDataset:
    name: str
    df: pd.DataFrame
    path: str = ""
    x_column: str = "row_index"
    y_column: str = ""
    cached_plot_xs: list[float] = field(default_factory=list)
    cached_plot_ys: list[float] = field(default_factory=list)
    cached_x_column: str = ""
    cached_y_column: str = ""
    cache_ready: bool = False


class SimImGuiApp:
    def __init__(self, backend: SimulationBackend):
        self.backend = backend
        self.ui = UIState()
        self.ui.run_speed_ms = 100

        self.window = None
        self.want_about_popup = False
        self.want_import_csv = False
        self._ctrl_a_latched = False
        self._ctrl_i_latched = False

        # Import stuff
        self.show_import_picker = False
        self.import_dir = os.getcwd()
        self.import_selected_files: list[str] = []
        self.show_import_format_window = False
        self.import_candidate_paths: list[str] = []
        self.import_file_headers: dict[str, list[str]] = {}
        self.import_merge_mode_options = ["keep_separate", "concat_rows", "merge_on_feature", "merge_on_posix_time"]
        self.import_merge_mode_idx = 0
        self.import_merge_how_options = ["inner", "left", "outer"]
        self.import_merge_how_idx = 0
        self.import_common_merge_columns: list[str] = []
        self.import_merge_column_idx = 0
        self.import_posix_time_column_idx = 0
        self.import_posix_unit_idx = 0 # this refers to the processing posix units, so 0=s, 1=ms, 2=us and 3=ns
        self.import_timezone_idx = 0
        self.last_import_summary: dict[str, Any] = {}

        # Modeling suite scren
        self.show_existing_model_picker = False
        self.model_picker_dir = os.getcwd()
        self.model_picker_selected: str | None = None
        self.show_csv_headers_window = False
        self.csv_headers: list[str] = []
        self.csv_path: str = ""
        self.feature_mask: list[bool] = []
        self.target_index: int = -1
        self.new_target_index: int = -1

        # Model family controls whether the observer renders a timing line plot
        # or a polarization/Poincare sphere.  The first option is intentionally
        # non-runnable so the user must choose timing or polarization before use.
        self.model_family_options = ["", "timing", "polarization"]
        self.model_family_labels = ["<select model family>", "Timing", "Polarization"]
        self.model_family_idx = 0
        self.active_model_family = ""

        self.model_type_options = ["default model", "existing model", "new model"]
        self.model_type_idx = 0
        self.existing_model_path = ""
        self.new_model_name = "my_model"
        self.new_epochs = 50
        self.new_lr = 1e-3
        self.model_kind_keys = list(SUPPORTED_MODEL_KINDS.keys())
        self.model_kind_labels = [SUPPORTED_MODEL_KINDS[key] for key in self.model_kind_keys]
        self.model_kind_idx = 0
        self.rf_n_estimators = 200
        self.rf_max_depth = 0
        self.last_training_summary: dict[str, Any] = {}
        self.split_train_pct = 70
        self.split_validation_pct = 15

        # Used for dataset type selection:
        self.observer_datasets: list[ObserverDataset] = []
        self.observer_selected_dataset_names: set[str] = set()
        self.observer_show_simulation = True

        # Processing data
        self.show_processing_window = False
        self.processing_datasets: list[ProcessingDataset] = []
        self.processing_active_dataset_idx = 0
        self.processing_selected_dataset_indices: set[int] = set()
        self.processing_selected_variables: set[str] = set()
        self.processing_timezones = list(COMMON_TIMEZONES)
        self.processing_posix_units = ["s", "ms", "us", "ns"]
        self.processing_quantify_methods = [
            ("rolling_extremal_difference", "Rolling extremal difference"),
            ("rolling_anchored_difference", "Rolling anchored difference"),
            ("rolling_stdev", "Rolling stdev"),
            ("rolling_variance", "Rolling variance"),
        ]
        self.processing_quantify_idx = 0
        self.processing_window_size = 10
        self.processing_poly_order = 2
        self.processing_interp_order = 1
        self.processing_fill_directions = ["forward", "backward", "both"]
        self.processing_fill_direction_idx = 0
        self.processing_downsample_methods = ["decimation", "avg", "max", "min", "filtered"]
        self.processing_downsample_idx = 0
        self.processing_downsample_window = 2
        self.processing_reference_dataset_idx = -1
        self.processing_interaction_name = ""
        self.processing_average_name = ""
        self.processing_merged_dataset_name = "merged_dataset"
        self.processing_stats_popup = False
        self.processing_stats_text = ""

        self.data_label = "Default example data"
        self.plot_label = "Photon Travel Time (seconds)"
        self.current_model_name = "default_channel_model"

        # added for image customization (these are just defaults until someone edits them):
        self.show_plot_settings_popup = False
        self.timing_plot_title = "Predicted Propagation Delay"
        self.timing_plot_title_font_size = 18.0
        self.timing_plot_x_axis_label = "Epoch"
        self.timing_plot_x_axis_font_size = 13.0
        self.timing_plot_y_axis_label = "Prediction"
        self.timing_plot_y_axis_font_size = 13.0
        self.timing_plot_tick_frequency = 0.0
        self.timing_plot_tick_font_size = 11.0
        # ADDED
        self.timing_plot_show_target = False
        self.timing_plot_target_y_axis_label = "Target Values"
        self.timing_plot_target_y_axis_font_size = 13.0
        self.timing_plot_x_axis_options = ["epoch", "posix_time"]
        self.timing_plot_x_axis_labels = ["Index / Epoch", "POSIX Time"]
        self.timing_plot_x_axis_idx = 0
        self.active_target_name = ""
        self.polarization_plot_title = "Poincare Sphere"
        self.polarization_plot_title_font_size = 18.0

    @property
    def selected_feature_names(self) -> list[str]:
        return [name for enabled, name in zip(self.feature_mask, self.csv_headers) if enabled]

    @property
    def selected_target_name(self) -> str | None:
        if 0 <= self.new_target_index < len(self.csv_headers):
            return self.csv_headers[self.new_target_index]
        return None

    @property
    def selected_model_kind(self) -> str:
        return self.model_kind_keys[self.model_kind_idx]

    @property
    def selected_model_family(self) -> str | None:
        if 0 <= self.model_family_idx < len(self.model_family_options):
            value = self.model_family_options[self.model_family_idx]
            return value or None
        return None

    @property
    def selected_model_family_label(self) -> str:
        family = self.selected_model_family
        if family == "timing":
            return "Timing line plot"
        if family == "polarization":
            return "Poincare/Bloch sphere"
        return "<none selected>"
    
    @property
    def selected_timing_x_axis_mode(self) -> str:
        if 0 <= self.timing_plot_x_axis_idx < len(self.timing_plot_x_axis_options):
            return self.timing_plot_x_axis_options[self.timing_plot_x_axis_idx]
        return "epoch"

    # ADDED
    def default_timing_x_axis_label(self) -> str:
        if self.selected_timing_x_axis_mode == "posix_time":
            return "POSIX Time"
        return "Epoch"

    # ADDED
    def set_timing_x_axis_idx(self, idx: int) -> None:
        old_default_labels = {"Epoch", "Index", "Index / Epoch", "POSIX Time"}
        old_label = str(self.timing_plot_x_axis_label or "")
        self.timing_plot_x_axis_idx = max(0, min(int(idx), len(self.timing_plot_x_axis_options) - 1))
        if not old_label.strip() or old_label in old_default_labels:
            self.timing_plot_x_axis_label = self.default_timing_x_axis_label()
    
    @property
    def split_test_pct(self) -> int:
        return max(0, 100 - int(self.split_train_pct) - int(self.split_validation_pct))
    
    def current_split_fractions(self) -> tuple[float, float, float]:
        train_pct = max(1, min(98, int(self.split_train_pct)))
        max_val = max(1, 99 - train_pct)
        validation_pct = max(1, min(max_val, int(self.split_validation_pct)))

        if train_pct + validation_pct >= 100:
            validation_pct = max(1, 99 - train_pct)

        self.split_train_pct = train_pct
        self.split_validation_pct = validation_pct
        test_pct = max(1, 100 - train_pct - validation_pct)

        return train_pct / 100.0, validation_pct / 100.0, test_pct / 100.0

    # TODO: split this into multiple files, maybe?
    def current_model_params(self) -> dict[str, Any]:
        if self.selected_model_kind == "random_forest":
            return {
                "n_estimators": int(self.rf_n_estimators),
                "max_depth": int(self.rf_max_depth),
                "random_state": 42,
            }
        return {}

    def set_status(self, text: str) -> None:
        with self.ui.lock:
            self.ui.status = text

    def _ask_save_path(self, *, title: str, default_filename: str, default_extension: str, filters: list[str], initial_dir: str | None = None) -> str:
        path = filedialpy.saveFile(
            initial_dir=initial_dir or os.getcwd(),
            initial_file=default_filename,
            title=title,
            filter=filters)

        if not path:
            return ""

        root, ext = os.path.splitext(path)
        if default_extension and not ext:
            path = f"{root}{default_extension}"

        return path

    def cb_plot(self, epoch: int, travel_time_s: float) -> None:
        with self.ui.lock:
            self.ui.epochs.append(epoch)
            self.ui.times.append(travel_time_s)

    def cb_conditions(self, conditions: dict[str, Any]) -> None:
        with self.ui.lock:
            self.ui.conditions = dict(conditions)

    def cb_poincare(self, state: Any) -> None:
        if state is None:
            return
        with self.ui.lock:
            self.ui.poincare_state = state
            if hasattr(self.ui, "poincare_states"):
                self.ui.poincare_states.append(state)
                if len(self.ui.poincare_states) > 500:
                    del self.ui.poincare_states[: len(self.ui.poincare_states) - 500]

    def _clear_plot_state(self) -> None:
        with self.ui.lock:
            self.ui.epochs.clear()
            self.ui.times.clear()
            self.ui.conditions.clear()
            self.ui.poincare_state = None
            if hasattr(self.ui, "poincare_states"):
                self.ui.poincare_states.clear()
            self.ui.running = False

    def _set_plot_label_for_config(self, config: ChannelModelConfig) -> None:
        family = str(config.model_family or "timing").strip().lower()
        self.active_model_family = family if family in {"timing", "polarization"} else "timing"

        self.active_target_name = str(config.target_name or "").strip()

        if self.active_model_family == "polarization":
            if config.mode == "default":
                self.plot_label = "Poincare Sphere: Polarization Random Walk"
                self.current_model_name = "polarization_random_walk"
            elif config.target_name:
                self.plot_label = f"Poincare Sphere: Predicted {config.target_name}"
                self.current_model_name = config.model_name or "polarization_model"
            else:
                self.plot_label = "Poincare Sphere: Current Polarization Model"
                self.current_model_name = config.model_name or "polarization_model"
            self.polarization_plot_title = self.plot_label
            return

        if config.mode == "default":
            self.plot_label = "Predicted Propagation Delay (seconds)"
            self.current_model_name = "default_physical_delay_model"
        elif config.mode == "existing":
            self.plot_label = f"Predicted {config.target_name}" if config.target_name else "Current Timing Model Output"
            self.current_model_name = config.model_name or "loaded_model"
        elif config.target_name:
            self.plot_label = f"Predicted {config.target_name}"
            self.current_model_name = config.model_name or "trained_model"
        else:
            self.plot_label = "Current Timing Model Output"
            self.current_model_name = config.model_name or "current_model"
        self.timing_plot_title = self.plot_label
        self.timing_plot_y_axis_label = config.target_name or "Prediction"

        if config.target_name:
            self.timing_plot_target_y_axis_label = f"Actual {config.target_name}"

    def generate(self, *, status_prefix: str = "Generated") -> None:
        if self.active_model_family not in {"timing", "polarization"}:
            self.set_status("Select a model family and apply a model before generating the observer plot.")
            return

        try:
            self.backend.reset()
            self._clear_plot_state()
            with self.ui.lock:
                self.ui.running = True
            self.backend.start(self.cb_plot, self.cb_conditions, self.cb_poincare)
            with self.ui.lock:
                point_count = len(self.ui.poincare_states) if self.active_model_family == "polarization" else len(self.ui.times)
                self.ui.running = False
            self.set_status(f"{status_prefix} {point_count} point(s) in {self.backend.get_mode_name()} mode.")
        except Exception as exc:
            with self.ui.lock:
                self.ui.running = False
            self.set_status(f"Generate failed: {exc}")

    # Keeping this so it doesnt break the other parts of the code (less for me to refactor, lol)
    def start(self) -> None:
        self.generate()

    def restart(self) -> None:
        self.generate(status_prefix="Regenerated")

    def stop(self) -> None:
        self.ui.running = False
        try:
            self.backend.stop()
            self.set_status("Simulation stopped.")
        except Exception as exc:
            self.set_status(f"Stop failed: {exc}")

    # helper functions
    def import_csv(self) -> None:
        self.show_import_picker = True
        self.import_selected_files = []
        self.show_import_format_window = False

    def _read_csv_headers(self, path: str) -> list[str]:
        headers = self._read_table(path).columns.tolist()
        return [str(h).strip() for h in headers if h is not None]

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = Path(path).suffix.lower()
        if ext not in {".csv", ".txt"}:
            raise ValueError("Only .csv and .txt files are supported.")
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        
    def _default_observer_y_column(self, df: pd.DataFrame) -> str:
        cols = [str(col) for col in df.columns]
        preferred_block = {"epoch", "posix_time", "row_index"}
        for col in cols:
            if str(col).lower() in preferred_block:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                return str(col)
        for col in cols:
            if str(col).lower() not in preferred_block:
                return str(col)
        return cols[0] if cols else ""

    def _default_observer_x_column(self, df: pd.DataFrame) -> str:
        for candidate in (POSIX_TIME_COL, "epoch"):
            if candidate in df.columns:
                return candidate
        time_candidates = candidate_time_columns(df)
        if time_candidates:
            return str(time_candidates[0])
        return "row_index"

    def _register_observer_dataset(self, name: str, df: pd.DataFrame, *, path: str = "") -> None:

        clean_name = name.strip() or f"dataset_{len(self.observer_datasets) + 1}"
        observer = ObserverDataset(name=clean_name, df=df, path=path, x_column=self._default_observer_x_column(df), y_column=self._default_observer_y_column(df))
        replaced = False
        for idx, existing in enumerate(self.observer_datasets):
            if existing.name == clean_name:
                was_selected = existing.name in self.observer_selected_dataset_names
                self.observer_datasets[idx] = observer
                if was_selected:
                    self.observer_selected_dataset_names.add(clean_name)
                replaced = True
                break
        if not replaced:
            self.observer_datasets.append(observer)

    def _invalidate_observer_dataset_cache(self, dataset: ObserverDataset) -> None:
        dataset.cached_plot_xs.clear()
        dataset.cached_plot_ys.clear()
        dataset.cached_x_column = ""
        dataset.cached_y_column = ""
        dataset.cache_ready = False

    def _build_observer_dataset_cache(self, dataset: ObserverDataset) -> tuple[list[float], list[float], str, str]:
        if dataset.df.empty:
            self._invalidate_observer_dataset_cache(dataset)
            return [], [], dataset.x_column, dataset.y_column

        df = dataset.df
        x_col = dataset.x_column if dataset.x_column in df.columns or dataset.x_column == "row_index" else self._default_observer_x_column(df)
        y_col = dataset.y_column if dataset.y_column in df.columns else self._default_observer_y_column(df)

        if x_col == "row_index":
            x_series = pd.Series(range(len(df)), dtype=float)
        else:
            x_series = pd.to_numeric(df[x_col], errors="coerce")

        y_series = pd.to_numeric(df[y_col], errors="coerce")
        valid_mask = x_series.notna() & y_series.notna()

        dataset.cached_plot_xs = x_series[valid_mask].astype(float).tolist()
        dataset.cached_plot_ys = y_series[valid_mask].astype(float).tolist()
        dataset.cached_x_column = x_col
        dataset.cached_y_column = y_col
        dataset.cache_ready = True

        return dataset.cached_plot_xs, dataset.cached_plot_ys, x_col, y_col

    def _load_csv_and_open_headers(self, path: str) -> None:
        self.backend.load_data(path)
        df = self._read_table(path)
        headers = [str(h).strip() for h in df.columns.tolist() if h is not None]
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        self._register_observer_dataset(dataset_name, df, path=path)
        self.csv_path = path
        self.csv_headers = headers
        self.feature_mask = [False] * len(headers)
        self.target_index = -1
        self.new_target_index = -1
        self.show_csv_headers_window = True
        self.data_label = f"CSV: {os.path.basename(path)}"
        self.last_training_summary = {}
        self._clear_plot_state()
        self.set_status(f"Loaded into modeling suite: {path}")

    def _start_import_formatting(self, paths: list[str]) -> None:
        if not paths:
            self.set_status("Select at least one data file first.")
            return
        
        self.import_candidate_paths = list(paths)
        self.import_file_headers = {path: self._read_csv_headers(path) for path in paths}
        header_sets = [set(headers) for headers in self.import_file_headers.values() if headers]
        self.import_common_merge_columns = sorted(set.intersection(*header_sets)) if header_sets else []
        self.import_merge_column_idx = 0
        self.import_merge_how_idx = 0
        self.import_posix_time_column_idx = 0
        self.import_posix_unit_idx = 0
        self.import_timezone_idx = 0

        if len(paths) <= 1:
            self.import_merge_mode_idx = 0
        elif self.import_common_merge_columns:
            self.import_merge_mode_idx = 0
        else:
            self.import_merge_mode_idx = 1
        self.show_import_format_window = True

    def _build_import_datasets(self) -> tuple[list[ProcessingDataset], dict[str, Any]]:

        paths = list(self.import_candidate_paths)
        if not paths:
            raise ValueError("No input files selected.")
        frames: list[tuple[str, pd.DataFrame]] = []
        for path in paths:
            frames.append((path, self._read_table(path)))
        source_rows = int(sum(len(df) for _, df in frames))
        merge_mode = self.import_merge_mode_options[self.import_merge_mode_idx]
        datasets: list[ProcessingDataset] = []

        if len(frames) == 1 or merge_mode == "keep_separate":
            for path, df in frames:
                datasets.append(
                    ProcessingDataset(
                        name=os.path.splitext(os.path.basename(path))[0],
                        df=df.copy(),
                        source_paths=[path],
                    )
                )
            combine_mode = "keep_separate" if len(frames) > 1 else "single_file"
        elif merge_mode == "concat_rows":
            combined = pd.concat([df for _, df in frames], ignore_index=True, sort=False)
            datasets = [
                ProcessingDataset(
                    name="concatenated_import",
                    df=combined,
                    source_paths=paths,
                )
            ]
            combine_mode = "concat_rows"
        elif merge_mode == "merge_on_posix_time":
            if not self.import_common_merge_columns:
                raise ValueError("No shared time column is available across the selected files.")

            time_col = self.import_common_merge_columns[self.import_posix_time_column_idx]
            posix_unit = self.processing_posix_units[self.import_posix_unit_idx]
            timezone = self.processing_timezones[self.import_timezone_idx]

            temp_datasets: list[ProcessingDataset] = []
            for path, df in frames:
                temp_datasets.append(
                    ProcessingDataset(
                        name=os.path.splitext(os.path.basename(path))[0],
                        df=df.copy(),
                        source_paths=[path],
                        time_column=time_col,
                        timezone=timezone,
                        posix_unit=posix_unit))

            merged = merge_datasets_on_posix(
                temp_datasets,
                merged_name="merged_import",
            )

            datasets = [merged]
            combine_mode = f"merge_on_posix_time:{posix_unit}"
        else:
            if not self.import_common_merge_columns:
                raise ValueError("No shared merge column is available across the selected files.")
            merge_col = self.import_common_merge_columns[self.import_merge_column_idx]
            merge_how = self.import_merge_how_options[self.import_merge_how_idx]
            df = frames[0][1].copy()
            suffix_counter = 2
            for _, next_df in frames[1:]:
                df = pd.merge(df, next_df, on=merge_col, how=merge_how, suffixes=("", f"_{suffix_counter}"))
                suffix_counter += 1
            datasets = [
                ProcessingDataset(
                    name="merged_import",
                    df=df,
                    source_paths=paths,
                )
            ]
            combine_mode = f"merge_on_feature:{merge_how}"

        summary = {
            "n_files": len(paths),
            "source_rows": source_rows,
            "n_datasets": len(datasets),
            "combine_mode": combine_mode,
            "paths": list(paths),
        }

        if len(paths) > 1 and merge_mode == "merge_on_feature" and self.import_common_merge_columns:
            summary["merge_column"] = self.import_common_merge_columns[self.import_merge_column_idx]
            summary["merge_how"] = self.import_merge_how_options[self.import_merge_how_idx]

        if len(paths) > 1 and merge_mode == "merge_on_posix_time" and self.import_common_merge_columns:
            summary["time_column"] = self.import_common_merge_columns[self.import_posix_time_column_idx]
            summary["posix_unit"] = self.processing_posix_units[self.import_posix_unit_idx]
            summary["timezone"] = self.processing_timezones[self.import_timezone_idx]

        return datasets, summary

    def confirm_import_format_and_load(self) -> None:
        try:
            datasets, summary = self._build_import_datasets()
            if not datasets:
                raise ValueError("No datasets were built from the selected files.")
            self.processing_datasets = datasets
            self.processing_active_dataset_idx = 0
            self.processing_selected_dataset_indices = {0}
            self.processing_selected_variables = set()
            self.show_processing_window = True
            self.show_import_format_window = False
            self.show_import_picker = False
            self.import_selected_files = []
            self.last_import_summary = summary
            self._sync_processing_selection()
            self.data_label = f"Processing suite: {len(datasets)} dataset(s) loaded"
            self.set_status(
                f"Loaded {len(datasets)} dataset(s) into the data processing suite from {summary.get('n_files', 1)} file(s)."
            )
        except Exception as exc:
            self.set_status(f"Import failed: {exc}")

    # Data processing helper funcs
    def _sync_processing_selection(self) -> None:
        if not self.processing_datasets:
            self.processing_active_dataset_idx = 0
            self.processing_selected_dataset_indices = set()
            self.processing_selected_variables = set()
            return
        self.processing_active_dataset_idx = max(0, min(self.processing_active_dataset_idx, len(self.processing_datasets) - 1))
        valid_indices = {idx for idx in self.processing_selected_dataset_indices if 0 <= idx < len(self.processing_datasets)}
        if not valid_indices:
            valid_indices = {self.processing_active_dataset_idx}
        self.processing_selected_dataset_indices = valid_indices
        active_ds = self.processing_datasets[self.processing_active_dataset_idx]
        active_cols = {str(col) for col in active_ds.df.columns}
        self.processing_selected_variables = {col for col in self.processing_selected_variables if col in active_cols}

    @property
    def active_processing_dataset(self) -> ProcessingDataset:
        self._sync_processing_selection()
        return self.processing_datasets[self.processing_active_dataset_idx]

    @property
    def selected_processing_datasets(self) -> list[ProcessingDataset]:
        self._sync_processing_selection()
        return [self.processing_datasets[idx] for idx in sorted(self.processing_selected_dataset_indices)]

    @property
    def selected_processing_variables(self) -> list[str]:
        self._sync_processing_selection()
        cols = list(self.active_processing_dataset.df.columns)
        return [col for col in cols if col in self.processing_selected_variables]

    def dataset_columns(self, dataset: ProcessingDataset) -> list[str]:
        return [str(col) for col in dataset.df.columns]

    def set_active_processing_dataset(self, idx: int) -> None:
        if 0 <= idx < len(self.processing_datasets):
            self.processing_active_dataset_idx = idx
            self.processing_selected_dataset_indices.add(idx)
            self._sync_processing_selection()

    def toggle_processing_dataset_selection(self, idx: int) -> None:
        if idx in self.processing_selected_dataset_indices:
            self.processing_selected_dataset_indices.remove(idx)
        else:
            self.processing_selected_dataset_indices.add(idx)
        if idx not in self.processing_selected_dataset_indices and self.processing_active_dataset_idx == idx:
            self.processing_active_dataset_idx = max(0, min(self.processing_active_dataset_idx, len(self.processing_datasets) - 1))
        self._sync_processing_selection()

    def toggle_processing_variable(self, name: str) -> None:
        if name in self.processing_selected_variables:
            self.processing_selected_variables.remove(name)
        else:
            self.processing_selected_variables.add(name)
        self._sync_processing_selection()

    def set_processing_time_metadata(self, dataset_idx: int, *, time_column: str | None = None, timezone: str | None = None, posix_unit: str | None = None) -> None:
        ds = self.processing_datasets[dataset_idx]

        if time_column is not None:
            ds.pending_time_column = time_column

        if timezone is not None:
            ds.pending_timezone = timezone

        if posix_unit is not None:
            ds.pending_posix_unit = posix_unit

    def apply_processing_time_metadata(self, dataset_idx: int) -> None:
        ds = self.processing_datasets[dataset_idx]

        if not ds.pending_time_column:
            raise ValueError("Select a time feature first.")

        if not ds.pending_timezone:
            raise ValueError("Select a timezone first.")

        if not ds.pending_posix_unit:
            raise ValueError("Select a POSIX unit first.")

        ds.time_column = ds.pending_time_column
        ds.timezone = ds.pending_timezone
        ds.posix_unit = ds.pending_posix_unit
        ds.posix_time_ready = False

        ensure_posix_time(ds)

        self.set_status(
            f"Applied POSIX time settings to '{ds.name}': "
            f"{ds.time_column}, {ds.timezone}, {ds.posix_unit}"
        )

    def _append_processing_dataset(self, dataset: ProcessingDataset, *, select: bool = True) -> None:
        self.processing_datasets.append(dataset)
        new_idx = len(self.processing_datasets) - 1
        if select:
            self.processing_active_dataset_idx = new_idx
            self.processing_selected_dataset_indices = {new_idx}
            self.processing_selected_variables = set()
        self._sync_processing_selection()

    def clear_all_processing_data(self) -> None:
        self.processing_datasets = []
        self.processing_selected_dataset_indices = set()
        self.processing_selected_variables = set()
        self.show_processing_window = False
        self.set_status("Cleared all data from the processing suite.")

    def exit_processing_window(self) -> None:
        self.show_processing_window = False
        self.set_status("Closed the data processing suite.")

    def _selected_dataset_count(self) -> int:
        return len(self.selected_processing_datasets)

    def _selected_variable_count(self) -> int:
        return len(self.selected_processing_variables)

    # TODO: Update this later to include horizonal scroll bar (the stats are cut off as-is)
    def show_processing_stats(self) -> None:
        try:
            self.processing_stats_text = descriptive_stats_text(self.active_processing_dataset)
            self.processing_stats_popup = True
        except Exception as exc:
            self.set_status(f"Stats failed: {exc}")

    def processing_take_derivative(self) -> None:
        try:
            vars_ = self.selected_processing_variables
            if len(vars_) != 1:
                raise ValueError("Select exactly one variable.")
            out_name = add_derivative(self.active_processing_dataset, vars_[0])
            self.set_status(f"Created derivative column '{out_name}'.")
        except Exception as exc:
            self.set_status(f"Derivative failed: {exc}")

    def processing_quantify_variable(self) -> None:
        try:
            vars_ = self.selected_processing_variables
            if len(vars_) != 1:
                raise ValueError("Select exactly one variable.")
            method = self.processing_quantify_methods[self.processing_quantify_idx][0]
            out_name = add_quantified_column(
                self.active_processing_dataset,
                vars_[0],
                method=method,
                window=max(1, int(self.processing_window_size)),
            )
            self.set_status(f"Created quantified column '{out_name}'.")
        except Exception as exc:
            self.set_status(f"Quantification failed: {exc}")

    def processing_remove_variables(self) -> None:
        try:
            vars_ = self.selected_processing_variables
            if len(vars_) < 1:
                raise ValueError("Select at least one variable.")
            remove_columns(self.active_processing_dataset, vars_)
            self.processing_selected_variables = set()
            self._sync_processing_selection()
            self.set_status(f"Removed {len(vars_)} variable(s) from '{self.active_processing_dataset.name}'.")
        except Exception as exc:
            self.set_status(f"Remove variable failed: {exc}")

    def processing_polynomial_expand(self) -> None:
        try:
            vars_ = self.selected_processing_variables
            if len(vars_) < 1:
                raise ValueError("Select one or more variables.")
            created = polynomial_expand(self.active_processing_dataset, vars_, degree=max(2, int(self.processing_poly_order)))
            self.set_status(f"Created {len(created)} polynomial feature(s).")
        except Exception as exc:
            self.set_status(f"Polynomial expansion failed: {exc}")

    def processing_drop_duplicate_timestamps(self) -> None:
        try:
            removed = remove_duplicate_timestamps(self.active_processing_dataset)
            self.set_status(f"Removed {removed} duplicate timestamp row(s).")
        except Exception as exc:
            self.set_status(f"Duplicate timestamp removal failed: {exc}")

    def processing_drop_nan_rows(self) -> None:
        try:
            removed = drop_nan_rows(self.active_processing_dataset, self.selected_processing_variables)
            self.set_status(f"Removed {removed} row(s) containing NaNs in the selected variables.")
        except Exception as exc:
            self.set_status(f"NaN row removal failed: {exc}")

    def processing_interpolate_missing(self) -> None:
        try:
            interpolate_missing(self.active_processing_dataset, self.selected_processing_variables, order=max(1, int(self.processing_interp_order)))
            self.set_status("Interpolated missing values for the selected variables.")
        except Exception as exc:
            self.set_status(f"Interpolation failed: {exc}")

    def processing_fill_missing(self) -> None:
        try:
            direction = self.processing_fill_directions[self.processing_fill_direction_idx]
            fill_missing(self.active_processing_dataset, self.selected_processing_variables, direction=direction)
            self.set_status(f"Filled missing values using {direction} fill.")
        except Exception as exc:
            self.set_status(f"Fill failed: {exc}")

    def processing_new_interaction_term(self) -> None:
        try:
            vars_ = self.selected_processing_variables
            if len(vars_) < 2:
                raise ValueError("Select at least two variables.")
            out_name = create_interaction_term(
                self.active_processing_dataset,
                vars_,
                output_name=self.processing_interaction_name.strip() or None,
            )
            self.set_status(f"Created interaction term '{out_name}'.")
        except Exception as exc:
            self.set_status(f"Interaction term failed: {exc}")

    def processing_merge_values_average(self) -> None:
        try:
            vars_ = self.selected_processing_variables
            if len(vars_) < 2:
                raise ValueError("Select at least two variables.")
            out_name = create_average_merge(
                self.active_processing_dataset,
                vars_,
                output_name=self.processing_average_name.strip() or None,
            )
            self.set_status(f"Created averaged feature '{out_name}'.")
        except Exception as exc:
            self.set_status(f"Merge values failed: {exc}")

    def processing_downsample_selected_sets(self) -> None:
        try:
            datasets = self.selected_processing_datasets
            if len(datasets) < 1:
                raise ValueError("Select at least one dataset.")
            method = self.processing_downsample_methods[self.processing_downsample_idx]
            reference = None
            if 0 <= self.processing_reference_dataset_idx < len(self.processing_datasets):
                reference = self.processing_datasets[self.processing_reference_dataset_idx]
            outputs: list[ProcessingDataset] = []
            for ds in datasets:
                ref = None if reference is None or reference is ds else reference
                outputs.append(
                    downsample_dataset(ds, method=method, window=max(1, int(self.processing_downsample_window)), reference=ref)
                )
            for out in outputs:
                self._append_processing_dataset(out, select=False)
            self.set_status(f"Created {len(outputs)} downsampled dataset(s) using {method}.")
        except Exception as exc:
            self.set_status(f"Downsample failed: {exc}")

    def processing_merge_selected_sets(self) -> None:
        try:
            source_count = len(self.selected_processing_datasets)
            merged = merge_datasets_on_posix(
                self.selected_processing_datasets,
                merged_name=self.processing_merged_dataset_name.strip() or "merged_dataset",
            )
            self._append_processing_dataset(merged)
            self.set_status(f"Merged {source_count} dataset(s) into '{merged.name}'.")
        except Exception as exc:
            self.set_status(f"Multi-set merge failed: {exc}")

    def save_active_processing_dataset(self) -> None:
        try:
            ds = self.active_processing_dataset
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in ds.name) or "dataset"

            path = self._ask_save_path(
                                    title="Save Active Dataset as CSV",
                                default_filename=f"{safe_name}.csv",
                            default_extension=".csv",
                        filters=["*.csv", "*"])

            if not path:
                self.set_status("Save dataframe cancelled.")
                return

            ds.df.to_csv(path, index=False)
            self.set_status(f"Saved dataframe to: {path}")
        except Exception as exc:
            self.set_status(f"Save dataframe failed: {exc}")

    def send_active_dataset_to_modeling(self) -> None:
        try:
            if len(self.processing_datasets) != 1:
                raise ValueError("Send to Modeling Suite requires exactly one dataset in the processing suite. Merge first if needed.")
            ds = self.processing_datasets[0]
            if ds.time_column is None:
                raise ValueError("Select a time feature before sending data to the modeling suite.")
            ensure_posix_time(ds)
            temp_dir = os.path.join(tempfile.gettempdir(), "simtwo_imports")
            os.makedirs(temp_dir, exist_ok=True)
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in ds.name) or "dataset"
            out_path = os.path.join(temp_dir, f"simtwo_{safe_name}_modeling.csv")
            ds.df.to_csv(out_path, index=False)
            self.show_processing_window = False
            self._load_csv_and_open_headers(out_path)
        except Exception as exc:
            self.set_status(f"Send to modeling failed: {exc}")


    def observer_dataset_by_name(self, name: str) -> ObserverDataset | None:
        for ds in self.observer_datasets:
            if ds.name == name:
                return ds
        return None

    @property
    def selected_observer_datasets(self) -> list[ObserverDataset]:
        selected: list[ObserverDataset] = []
        for ds in self.observer_datasets:
            if ds.name in self.observer_selected_dataset_names:
                selected.append(ds)
        return selected

    def toggle_observer_dataset(self, name: str) -> None:
        if name in self.observer_selected_dataset_names:
            self.observer_selected_dataset_names.remove(name)
        else:
            self.observer_selected_dataset_names.add(name)

    def set_observer_dataset_y_column(self, name: str, column: str) -> None:
        ds = self.observer_dataset_by_name(name)
        if ds is not None and ds.y_column != column:
            ds.y_column = column
            self._invalidate_observer_dataset_cache(ds)

    def set_observer_dataset_x_column(self, name: str, column: str) -> None:
        ds = self.observer_dataset_by_name(name)
        if ds is not None and ds.x_column != column:
            ds.x_column = column
            self._invalidate_observer_dataset_cache(ds)

    def observer_dataset_plot_series(self, dataset: ObserverDataset) -> tuple[list[float], list[float], str, str]:
        if (dataset.cache_ready and dataset.cached_x_column == dataset.x_column and dataset.cached_y_column == dataset.y_column):
            return dataset.cached_plot_xs, dataset.cached_plot_ys, dataset.cached_x_column, dataset.cached_y_column

        return self._build_observer_dataset_cache(dataset)
    
    # ADDED
    def active_observer_dataset(self) -> ObserverDataset | None:
        if self.csv_path:
            active_name = os.path.splitext(os.path.basename(self.csv_path))[0]
            ds = self.observer_dataset_by_name(active_name)
            if ds is not None:
                return ds
        if self.observer_datasets:
            return self.observer_datasets[-1]
        return None
    
    def _timing_posix_column(self, dataset: ObserverDataset | None = None) -> str:
        dataset = dataset or self.active_observer_dataset()
        if dataset is None or dataset.df.empty:
            return ""
        for candidate in (POSIX_TIME_COL, "posix_time"):
            if candidate in dataset.df.columns:
                series = pd.to_numeric(dataset.df[candidate], errors="coerce")
                if series.notna().sum() >= 2:
                    return candidate
        return ""

    # ADDED
    def timing_posix_x_available(self) -> bool:
        return bool(self._timing_posix_column())

    # ADDED
    def timing_x_axis_status(self) -> str:
        if self.timing_posix_x_available():
            dataset = self.active_observer_dataset()
            column = self._timing_posix_column(dataset)
            return f"POSIX x-axis available from '{column}' in '{dataset.name}'." if dataset is not None else "POSIX x-axis available."
        return "No numeric POSIX time feature is available; index/epoch x-axis will be used."

    # ADDED
    def current_timing_plot_xs(self, epochs: list[int]) -> list[float]:
        if self.selected_timing_x_axis_mode != "posix_time":
            return [float(epoch) for epoch in epochs]

        dataset = self.active_observer_dataset()
        column = self._timing_posix_column(dataset)
        if dataset is None or not column:
            return [float(epoch) for epoch in epochs]

        series = pd.to_numeric(dataset.df[column], errors="coerce")
        xs: list[float] = []
        for epoch in epochs:
            try:
                value = series.iloc[int(epoch)]
            except Exception:
                value = float("nan")
            if pd.notna(value):
                xs.append(float(value))
            else:
                xs.append(float(epoch))
        return xs

    # ADDED
    def _timing_target_column_candidates(self, dataset: ObserverDataset | None = None) -> list[str]:
        candidates: list[str] = []
        for value in (self.active_target_name, self.selected_target_name):
            if value and value not in candidates:
                candidates.append(str(value))
        preferred = [
            "path_delay",
            "propagation_delay",
            "prop_delay",
            "time_sync_error",
            "clock_error",
            "delay",
            "target",
        ]
        for value in preferred:
            if value not in candidates:
                candidates.append(value)
        if dataset is not None and dataset.y_column and dataset.y_column not in candidates:
            candidates.append(dataset.y_column)
        return candidates

    # ADDED
    def _available_timing_target_column(self) -> tuple[ObserverDataset | None, str]:
        dataset = self.active_observer_dataset()
        if dataset is None or dataset.df.empty:
            return None, ""
        lookup = {str(col).strip().lower(): str(col) for col in dataset.df.columns}
        for candidate in self._timing_target_column_candidates(dataset):
            column = lookup.get(str(candidate).strip().lower())
            if not column:
                continue
            series = pd.to_numeric(dataset.df[column], errors="coerce")
            if series.notna().sum() >= 2:
                return dataset, column
        return dataset, ""

    # ADDED
    def timing_target_overlay_status(self) -> tuple[bool, str]:
        dataset, column = self._available_timing_target_column()
        if dataset is None:
            return False, "Load a dataset with a numeric target column to enable target overlay."
        if column:
            return True, f"Target overlay available from '{column}' in '{dataset.name}'."
        if self.active_target_name:
            return False, f"No numeric '{self.active_target_name}' column is available in '{dataset.name}'."
        return False, f"No numeric timing target column is available in '{dataset.name}'."

    # ADDED
    def current_timing_target_overlay(self, xs: list[float]) -> tuple[list[float], list[float], str]:
        if not self.timing_plot_show_target:
            return [], [], ""
        dataset, column = self._available_timing_target_column()
        if dataset is None or not column or len(xs) < 1:
            return [], [], ""
        series = pd.to_numeric(dataset.df[column], errors="coerce")
        count = min(len(xs), len(series))
        target_xs: list[float] = []
        target_ys: list[float] = []
        for idx in range(count):
            value = series.iloc[idx]
            if pd.notna(value):
                target_xs.append(float(xs[idx]))
                target_ys.append(float(value))
        if len(target_xs) < 2:
            return [], [], ""
        return target_xs, target_ys, column

    # Helpers for modeling
    def export_results(self) -> None:
        try:
            path = self._ask_save_path(
                                    title="Export Results as CSV",
                                default_filename="results.csv",
                            default_extension=".csv",
                        filters=["*.csv", "*"])

            if not path:
                self.set_status("Export cancelled.")
                return

            self.backend.export_results(path)
            self.set_status(f"Exported: {path}")
        except Exception as exc:
            self.set_status(f"Export failed: {exc}")


    def save_current_plot(self) -> None:

        try:
            if self.active_model_family == "polarization":
                default_filename = "poincare_plot.png"
            else:
                default_filename = "timing_plot.png"

            path = self._ask_save_path(
                                    title="Save Plot Image",
                                default_filename=default_filename,
                            default_extension=".png",
                        filters=["*.png", "*.pdf", "*.svg", "*"])

            if not path:
                self.set_status("Plot save cancelled")
                return

            with self.ui.lock:
                xs = list(self.ui.epochs)
                ys = list(self.ui.times)
                poincare_states = list(getattr(self.ui, "poincare_states", []))
                if not poincare_states and getattr(self.ui, "poincare_state", None) is not None:
                    poincare_states = [self.ui.poincare_state]

            if self.active_model_family == "polarization":
                save_poincare_plot(
                    path,
                    poincare_states,
                    title=self.polarization_plot_title,
                    title_font_size=self.polarization_plot_title_font_size,
                )
            else:
                plot_xs = self.current_timing_plot_xs(xs)
                target_xs, target_ys, target_column = self.current_timing_target_overlay(plot_xs)
                save_timing_plot(
                    path,
                    plot_xs,
                    ys,
                    title=self.timing_plot_title,
                    title_font_size=self.timing_plot_title_font_size,
                    x_axis_label=self.timing_plot_x_axis_label,
                    x_axis_font_size=self.timing_plot_x_axis_font_size,
                    y_axis_label=self.timing_plot_y_axis_label,
                    y_axis_font_size=self.timing_plot_y_axis_font_size,
                    tick_frequency=self.timing_plot_tick_frequency,
                    tick_font_size=self.timing_plot_tick_font_size,
                    target_xs=target_xs,
                    target_ys=target_ys,
                    target_label=f"Actual {target_column}" if target_column else "Target",
                    target_y_axis_label=self.timing_plot_target_y_axis_label,
                    target_y_axis_font_size=self.timing_plot_target_y_axis_font_size,
                )

            self.set_status(f"Saved plot to: {path}")
        except Exception as exc:
            self.set_status(f"Save plot failed with this error: {exc}")


    def apply_model_config(self, config: ChannelModelConfig) -> None:
        if config.model_family not in {"timing", "polarization"}:
            self.set_status("Select a model family first: timing or polarization.")
            return
        try:
            self.backend.configure_channel_model(config)
            self.backend.reset()
            self._clear_plot_state()
            self._set_plot_label_for_config(config)
            self.set_status(f"Applied {config.model_family} model config '{config.model_name}' ({config.mode})")
        except Exception as exc:
            self.set_status(f"Model config failed: {exc}")

    def train_and_activate_model(self) -> None:
        family = self.selected_model_family
        if family is None:
            self.set_status("Select a model family first: timing or polarization.")
            return
        train_fraction, validation_fraction, test_fraction = self.current_split_fractions()
        config = ChannelModelConfig(
            mode="new",
            model_family=family,
            model_name=self.new_model_name.strip() or "my_model",
            epochs=int(self.new_epochs),
            learning_rate=float(self.new_lr),
            feature_names=self.selected_feature_names,
            target_name=self.selected_target_name,
            model_kind=self.selected_model_kind,
            model_params=self.current_model_params(),
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        try:
            summary = dict(self.backend.train_channel_model(config) or {})
            self.last_training_summary = summary

            active_name = os.path.splitext(os.path.basename(self.csv_path))[0] if self.csv_path else "active_dataset"
            active_ds = self.observer_dataset_by_name(active_name)
            if active_ds is not None and config.target_name:
                if config.target_name in active_ds.df.columns:
                    active_ds.y_column = config.target_name

            self.backend.reset()
            self._clear_plot_state()
            self._set_plot_label_for_config(config)
            self.set_status(
                f"Trained {config.model_family} {SUPPORTED_MODEL_KINDS.get(config.model_kind, config.model_kind)} as '{config.model_name}'. "
                f"Train RMSE={summary.get('train_rmse', float('nan')):.4g}, "
                f"Val RMSE={summary.get('validation_rmse', float('nan')):.4g}, "
                f"Test RMSE={summary.get('test_rmse', float('nan')):.4g}"
            )
        except Exception as exc:
            self.set_status(f"Training failed: {exc}")

    def save_current_model(self) -> None:
        safe_name = self.new_model_name.strip() or "trained_model"
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in safe_name).strip("_") or "trained_model"

        try:
            model_dir = os.path.join(os.getcwd(), "generated_models")
            os.makedirs(model_dir, exist_ok=True)

            path = self._ask_save_path(
                                    title="Save Trained Model",
                                default_filename=f"{safe_name}.joblib",
                            default_extension=".joblib",
                        filters=["*.joblib", "*"],
                    initial_dir=model_dir)

            if not path:
                self.set_status("Model save cancelled.")
                return

            self.backend.save_current_model(path)
            self.existing_model_path = path
            self.set_status(f"Saved trained model to: {path}")
        except Exception as exc:
            self.set_status(f"Save model failed: {exc}")

    # others helpers (drawing, etc)
    def _shortcut_ctrl_a(self) -> bool:
        if self.window is None:
            return False
        ctrl = (
            glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
            or glfw.get_key(self.window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
        )
        a_key = glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS
        return ctrl and a_key

    def _shortcut_ctrl_i(self) -> bool:
        if self.window is None:
            return False
        ctrl = (
            glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
            or glfw.get_key(self.window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
        )
        i_key = glfw.get_key(self.window, glfw.KEY_I) == glfw.PRESS
        return ctrl and i_key

    def _model_type(self) -> str:
        return self.model_type_options[self.model_type_idx]

    def close_window(self) -> None:
        if self.window is not None and glfw is not None:
            glfw.set_window_should_close(self.window, True)

    def draw_model_config_panel(self, width: int, height: int) -> None:
        draw_model_config_panel(self, width, height)

    def draw(self) -> None:
        draw_menu_bar(self)

        if self._shortcut_ctrl_a():
            if not self._ctrl_a_latched:
                self.want_about_popup = True
                self._ctrl_a_latched = True
        else:
            self._ctrl_a_latched = False

        if self._shortcut_ctrl_i():
            if not self._ctrl_i_latched:
                self.want_import_csv = True
                self._ctrl_i_latched = True
        else:
            self._ctrl_i_latched = False

        if self.want_import_csv:
            self.want_import_csv = False
            self.import_csv()

        imgui.set_next_window_position(10, 30)
        imgui.set_next_window_size(1120, 520)
        imgui.begin(
            "Simulation",
            flags=(
                imgui.WINDOW_NO_TITLE_BAR
                | imgui.WINDOW_NO_RESIZE
                | imgui.WINDOW_NO_MOVE
                | imgui.WINDOW_NO_COLLAPSE
            ),
        )
        right_panel_width = 360.0
        total_width = imgui.get_window_width()
        left_panel_width = max(300.0, total_width - right_panel_width - 36.0)

        imgui.columns(2, "main_cols", border=True)
        imgui.set_column_width(0, left_panel_width)

        draw_left_panel(self)
        imgui.next_column()
        draw_right_panel(self)

        imgui.columns(1)
        imgui.end()

        # Draw overlays after the base win so they cant get stuck behind it
        draw_about_popup(self)
        draw_import_csv_picker_window(self)
        draw_import_format_window(self)
        draw_data_processing_window(self)
        draw_csv_headers_window(self)


def run_app(backend: SimulationBackend) -> None:
    # TODO: Can probably remove since the app won t run without
    if glfw is None or imgui is None or GlfwRenderer is None or GL is None:
        # update: Do I still need these? Or does the toml file ensure that everything is installed? This might be superfluous, if so:
        raise RuntimeError("The ImGui GUI requires 'pyimgui', 'glfw', and 'PyOpenGL'. Install them before running the GUI.")

    glfw.init()

    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    window = glfw.create_window(1150, 600, "simtwo - ImGui GUI", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    imgui.create_context()
    impl = GlfwRenderer(window)

    app = SimImGuiApp(backend)
    app.window = window

    # main app loop:
    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            impl.process_inputs()

            imgui.new_frame()
            app.draw()
            imgui.render()

            GL.glClearColor(0.08, 0.08, 0.08, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            impl.render(imgui.get_draw_data())
            glfw.swap_buffers(window)
    finally:
        app.stop()
        impl.shutdown()
        glfw.terminate()
