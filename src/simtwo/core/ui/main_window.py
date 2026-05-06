from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

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

import glfw 
import imgui 
import pandas as pd
from imgui.integrations.glfw import GlfwRenderer 
from OpenGL import GL

import filedialpy


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
        with self.ui.lock:
            self.ui.poincare_state = state

    def _clear_plot_state(self) -> None:
        with self.ui.lock:
            self.ui.epochs.clear()
            self.ui.times.clear()
            self.ui.conditions.clear()
            self.ui.poincare_state = None
            self.ui.running = False

    def _set_plot_label_for_config(self, config: ChannelModelConfig) -> None:
        if config.mode == "default":
            self.plot_label = "Photon Travel Time (seconds)"
        elif config.mode == "existing":
            self.plot_label = "Current Model Output"
        elif config.target_name:
            self.plot_label = f"Predicted {config.target_name}"
        else:
            self.plot_label = "Current Model Output"

    def start(self) -> None:
        self.backend.set_run_speed(self.ui.run_speed_ms)
        self.ui.running = True
        try:
            self.backend.start(self.cb_plot, self.cb_conditions, self.cb_poincare)
            self.set_status(f"Running in {self.backend.get_mode_name()} mode.")
        except Exception as exc:
            self.ui.running = False
            self.set_status(f"Start failed: {exc}")

    def restart(self) -> None:
        try:
            self.backend.reset()
            self.ui.reset()
            self.start()
        except Exception as exc:
            self.set_status(f"Restart failed: {exc}")

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

    def _load_csv_and_open_headers(self, path: str) -> None:
        self.backend.load_data(path)
        headers = self._read_csv_headers(path)
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

    def apply_model_config(self, config: ChannelModelConfig) -> None:
        try:
            self.backend.configure_channel_model(config)
            self.backend.reset()
            self._clear_plot_state()
            self._set_plot_label_for_config(config)
            self.set_status(f"Applied model config '{config.model_name}' ({config.mode})")
        except Exception as exc:
            self.set_status(f"Model config failed: {exc}")

    def train_and_activate_model(self) -> None:
        config = ChannelModelConfig(
            mode="new",
            model_name=self.new_model_name.strip() or "my_model",
            epochs=int(self.new_epochs),
            learning_rate=float(self.new_lr),
            feature_names=self.selected_feature_names,
            target_name=self.selected_target_name,
            model_kind=self.selected_model_kind,
            model_params=self.current_model_params(),
        )
        try:
            summary = dict(self.backend.train_channel_model(config) or {})
            self.last_training_summary = summary
            self.backend.reset()
            self._clear_plot_state()
            self._set_plot_label_for_config(config)
            self.set_status(
                f"Trained {SUPPORTED_MODEL_KINDS.get(config.model_kind, config.model_kind)} as '{config.model_name}'. "
                f"RMSE={summary.get('train_rmse', float('nan')):.4g}, R²={summary.get('train_r2', float('nan')):.4g}"
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
        imgui.columns(2, "main_cols", border=True)
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
    # TODO: Can probably remove
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
