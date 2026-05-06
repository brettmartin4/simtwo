from __future__ import annotations

import os
from typing import TYPE_CHECKING


import imgui
from typing import TYPE_CHECKING

from simtwo.core.ui.data_processing import POSIX_TIME_COL, candidate_time_columns

if TYPE_CHECKING:
    from simtwo.core.ui.main_window import SimImGuiApp


MODEL_EXTENSIONS = (".json", ".pkl", ".pickle", ".joblib", ".keras", ".h5")
DATA_EXTENSIONS = (".csv", ".txt")


def _begin_fullscreen_window(name: str):
    vp = imgui.get_main_viewport()
    x, y = vp.pos
    w, h = vp.size
    menu_h = imgui.get_frame_height()
    imgui.set_next_window_position(x, y + menu_h)
    imgui.set_next_window_size(w, max(1.0, h - menu_h))
    flags = (
        imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_NO_COLLAPSE
        | imgui.WINDOW_NO_SAVED_SETTINGS
    )
    return imgui.begin(name, flags=flags)


def draw_about_popup(app: "SimImGuiApp") -> None:
    if app.want_about_popup:
        imgui.open_popup("About")
        app.want_about_popup = False

    opened, _ = imgui.begin_popup_modal("About", True)
    if opened:
        imgui.text("simtwo - Quantum Simulation GUI")
        imgui.separator()
        imgui.text(f"Backend mode: {app.backend.get_mode_name()}")
        imgui.text("Built with pyimgui + GLFW")
        imgui.spacing()
        if imgui.button("Close"):
            imgui.close_current_popup()
        imgui.end_popup()


def _toggle_selected_file(app: "SimImGuiApp", name: str) -> None:
    if name in app.import_selected_files:
        app.import_selected_files.remove(name)
    else:
        app.import_selected_files.append(name)


def draw_import_csv_picker_window(app: "SimImGuiApp") -> None:
    if not app.show_import_picker:
        return

    _begin_fullscreen_window("##ImportCSVFullscreen")
    imgui.text("Import Data Files")
    imgui.separator()
    imgui.text_wrapped("Load one or more CSV or text files. The next screen lets you decide whether to keep multiple files separate, concatenate them, or merge them by a shared feature before opening the data processing suite.")
    imgui.spacing()

    if not os.path.isdir(app.import_dir):
        app.import_dir = os.getcwd()

    imgui.text("Directory:")
    imgui.same_line()
    imgui.text(app.import_dir)
    imgui.separator()

    if imgui.button("Up", width=100):
        parent = os.path.dirname(app.import_dir)
        if parent and os.path.isdir(parent):
            app.import_dir = parent
            app.import_selected_files = []
    imgui.same_line()
    imgui.button("Refresh", width=100)
    imgui.same_line()
    if imgui.button("Clear Selection", width=160):
        app.import_selected_files = []

    avail_w, avail_h = imgui.get_content_region_available()
    list_h = max(180, int(avail_h - 180))

    try:
        entries = os.listdir(app.import_dir)
    except Exception as exc:
        app.set_status(f"List dir error: {exc}")
        entries = []

    folders = sorted([x for x in entries if os.path.isdir(os.path.join(app.import_dir, x))])
    data_files = sorted([x for x in entries if os.path.isfile(os.path.join(app.import_dir, x)) and x.lower().endswith(DATA_EXTENSIONS)])

    imgui.begin_child("##import_filelist", width=0, height=list_h, border=True)
    for name in folders:
        clicked, _ = imgui.selectable(f"[DIR] {name}", False)
        if clicked:
            app.import_dir = os.path.join(app.import_dir, name)
            app.import_selected_files = []
    for name in data_files:
        selected = name in app.import_selected_files
        clicked, _ = imgui.selectable(name, selected)
        if clicked:
            _toggle_selected_file(app, name)
    imgui.end_child()

    imgui.spacing()
    imgui.text(f"Selected files: {len(app.import_selected_files)}")
    if app.import_selected_files:
        imgui.begin_child("##selected_import_files", width=0, height=90, border=True)
        for name in app.import_selected_files:
            imgui.bullet_text(name)
        imgui.end_child()

    imgui.spacing()
    imgui.separator()
    can_continue = len(app.import_selected_files) > 0
    if not can_continue:
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
    clicked_next = imgui.button("Next", width=160)
    if not can_continue:
        imgui.pop_style_var()
    if clicked_next and can_continue:
        paths = [os.path.join(app.import_dir, item) for item in app.import_selected_files]
        app._start_import_formatting(paths)
        app.show_import_picker = False

    imgui.same_line()
    if imgui.button("Exit Import", width=160):
        app.show_import_picker = False
        app.import_selected_files = []

    imgui.end()


def draw_import_format_window(app: "SimImGuiApp") -> None:
    if not app.show_import_format_window:
        return

    _begin_fullscreen_window("##ImportFormatFullscreen")
    imgui.text("Import Formatting")
    imgui.separator()
    imgui.text_wrapped("This step only controls how the selected files enter the data processing suite. Detailed time normalization, resampling, derivative generation, merging by POSIX time, NaN handling, and export all happen inside the processing suite after this.")
    imgui.spacing()

    selected_paths = list(app.import_candidate_paths)
    imgui.text(f"Files selected: {len(selected_paths)}")
    imgui.begin_child("##import_candidates", width=0, height=110, border=True)
    for path in selected_paths:
        imgui.bullet_text(os.path.basename(path))
    imgui.end_child()

    if len(selected_paths) > 1:
        imgui.spacing()
        imgui.separator()
        imgui.text("Initial Multi-file Handling")

        labels = {
            "keep_separate": "Keep as separate datasets",
            "concat_rows": "Concatenate rows now",
            "merge_on_feature": "Merge by common feature now",
            "merge_on_posix_time": "Merge by POSIX time now"
        }

        current_key = app.import_merge_mode_options[app.import_merge_mode_idx]

        if imgui.begin_combo("Combine Method", labels[current_key]):
            for idx, key in enumerate(app.import_merge_mode_options):
                needs_common_column = key in ("merge_on_feature", "merge_on_posix_time")
                enabled = not (needs_common_column and not app.import_common_merge_columns)

                if not enabled:
                    imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

                selected = idx == app.import_merge_mode_idx
                clicked, _ = imgui.selectable(
                    labels[key],
                    selected,
                    0 if enabled else imgui.SELECTABLE_DISABLED,
                )

                if clicked and enabled:
                    app.import_merge_mode_idx = idx

                if selected:
                    imgui.set_item_default_focus()

                if not enabled:
                    imgui.pop_style_var()

            imgui.end_combo()

        merge_mode = app.import_merge_mode_options[app.import_merge_mode_idx]

        if merge_mode == "merge_on_feature":
            if not app.import_common_merge_columns:
                imgui.text_colored(
                    "No common headers were found across all selected files.",
                    1.0,
                    0.6,
                    0.4,
                )
            else:
                if app.import_merge_column_idx >= len(app.import_common_merge_columns):
                    app.import_merge_column_idx = 0

                current_col = app.import_common_merge_columns[app.import_merge_column_idx]

                if imgui.begin_combo("Common Feature", current_col):
                    for idx, col in enumerate(app.import_common_merge_columns):
                        selected = idx == app.import_merge_column_idx
                        clicked, _ = imgui.selectable(col, selected)

                        if clicked:
                            app.import_merge_column_idx = idx

                        if selected:
                            imgui.set_item_default_focus()

                    imgui.end_combo()

                merge_how_labels = {
                    "inner": "Inner join",
                    "left": "Left join",
                    "outer": "Outer join"
                }

                current_join = app.import_merge_how_options[app.import_merge_how_idx]

                if imgui.begin_combo("Join Type", merge_how_labels[current_join]):
                    for idx, key in enumerate(app.import_merge_how_options):
                        selected = idx == app.import_merge_how_idx
                        clicked, _ = imgui.selectable(merge_how_labels[key], selected)

                        if clicked:
                            app.import_merge_how_idx = idx

                        if selected:
                            imgui.set_item_default_focus()

                    imgui.end_combo()

        elif merge_mode == "merge_on_posix_time":
            if not app.import_common_merge_columns:
                imgui.text_colored(
                    "No common headers were found across all selected files.",
                    1.0,
                    0.6,
                    0.4,
                )
            else:
                imgui.spacing()
                imgui.text("POSIX Time Merge Settings")
                imgui.text_wrapped(
                    "Choose the shared time column and the unit used by those POSIX "
                    "timestamps before the files are merged."
                )

                if app.import_posix_time_column_idx >= len(app.import_common_merge_columns):
                    app.import_posix_time_column_idx = 0

                current_time_col = app.import_common_merge_columns[app.import_posix_time_column_idx]

                if imgui.begin_combo("Time Feature", current_time_col):
                    for idx, col in enumerate(app.import_common_merge_columns):
                        selected = idx == app.import_posix_time_column_idx
                        clicked, _ = imgui.selectable(col, selected)

                        if clicked:
                            app.import_posix_time_column_idx = idx

                        if selected:
                            imgui.set_item_default_focus()

                    imgui.end_combo()

                if app.import_posix_unit_idx >= len(app.processing_posix_units):
                    app.import_posix_unit_idx = 0

                current_unit = app.processing_posix_units[app.import_posix_unit_idx]

                if imgui.begin_combo("POSIX Unit", current_unit):
                    for idx, unit in enumerate(app.processing_posix_units):
                        selected = idx == app.import_posix_unit_idx
                        clicked, _ = imgui.selectable(unit, selected)

                        if clicked:
                            app.import_posix_unit_idx = idx

                        if selected:
                            imgui.set_item_default_focus()

                    imgui.end_combo()

                if app.import_timezone_idx >= len(app.processing_timezones):
                    app.import_timezone_idx = 0

                current_tz = app.processing_timezones[app.import_timezone_idx]

                if imgui.begin_combo("Timezone", current_tz):
                    for idx, tz in enumerate(app.processing_timezones):
                        selected = idx == app.import_timezone_idx
                        clicked, _ = imgui.selectable(tz, selected)

                        if clicked:
                            app.import_timezone_idx = idx

                        if selected:
                            imgui.set_item_default_focus()

                    imgui.end_combo()

    imgui.spacing()
    imgui.separator()
    imgui.text_wrapped("Next, the data processing suite will open. Each dataset there must be assigned a time feature, timezone, and POSIX unit before any time-aware operations like derivatives or multi-set merging will run.")

    imgui.spacing()

    if imgui.button("Load Into Data Processing Suite", width=260):
        app.confirm_import_format_and_load()

    imgui.same_line()

    if imgui.button("Back", width=120):
        app.show_import_format_window = False
        app.show_import_picker = True

    imgui.same_line()
    
    if imgui.button("Exit Import", width=160):
        app.show_import_format_window = False
        app.import_candidate_paths = []

    imgui.end()


def draw_existing_model_picker_window(app: "SimImGuiApp") -> None:
    if not app.show_existing_model_picker:
        return

    _begin_fullscreen_window("##ModelPickerFullscreen")
    imgui.text("Choose Model File")
    imgui.separator()

    if not os.path.isdir(app.model_picker_dir):
        app.model_picker_dir = os.getcwd()

    imgui.text("Directory:")
    imgui.same_line()
    imgui.text(app.model_picker_dir)

    if imgui.button("Up", width=100):
        parent = os.path.dirname(app.model_picker_dir)
        if parent and os.path.isdir(parent):
            app.model_picker_dir = parent
            app.model_picker_selected = None
    imgui.same_line()
    imgui.button("Refresh", width=100)

    try:
        entries = os.listdir(app.model_picker_dir)
    except Exception as exc:
        app.set_status(f"Model picker error: {exc}")
        entries = []

    folders = sorted([x for x in entries if os.path.isdir(os.path.join(app.model_picker_dir, x))])
    files = sorted(
        [x for x in entries if os.path.isfile(os.path.join(app.model_picker_dir, x)) and x.lower().endswith(MODEL_EXTENSIONS)]
    )

    imgui.begin_child("##model_file_list", width=0, height=300, border=True)
    for name in folders:
        clicked, _ = imgui.selectable(f"[DIR] {name}", False)
        if clicked:
            app.model_picker_dir = os.path.join(app.model_picker_dir, name)
            app.model_picker_selected = None
    for name in files:
        selected = app.model_picker_selected == name
        clicked, _ = imgui.selectable(name, selected)
        if clicked:
            app.model_picker_selected = name
    imgui.end_child()

    imgui.spacing()
    can_select = app.model_picker_selected is not None
    if not can_select:
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
    clicked_select = imgui.button("Select", width=160)
    if not can_select:
        imgui.pop_style_var()
    if clicked_select and can_select:
        app.existing_model_path = os.path.join(app.model_picker_dir, app.model_picker_selected)
        app.show_existing_model_picker = False

    imgui.same_line()
    if imgui.button("Exit", width=160):
        app.show_existing_model_picker = False
        app.model_picker_selected = None

    imgui.end()

# TODO: Consider maybe splitting this up so it isn't one massive monolith
def draw_data_processing_window(app: "SimImGuiApp") -> None:
    if not app.show_processing_window:
        return
    if not app.processing_datasets:
        imgui.open_popup("Data Processing Error")
        return

    _begin_fullscreen_window("##DataProcessingFullscreen")
    imgui.text("Data Processing Suite")
    imgui.separator()
    imgui.text_wrapped("Before the modeling suite can use the data, each dataset here must have a selected time feature, timezone, and POSIX unit. Time-aware operations use the standardized 'posix_time' column that gets generated from those settings.")
    imgui.spacing()

    avail_w, avail_h = imgui.get_content_region_available()
    left_w = 250
    mid_w = 290
    right_w = max(320, int(avail_w - left_w - mid_w - 24))
    panel_h = max(220, int(avail_h - 150))

    # Dataset panel
    imgui.begin_child("##processing_datasets", width=left_w, height=panel_h, border=True)
    imgui.text("Datasets")
    imgui.separator()
    for idx, ds in enumerate(app.processing_datasets):
        selected = idx in app.processing_selected_dataset_indices
        changed, value = imgui.checkbox(f"##dscheck_{idx}", selected)
        if changed:
            app.toggle_processing_dataset_selection(idx)
        imgui.same_line()
        clicked, _ = imgui.selectable(f"{ds.name} ({len(ds.df)} rows)", idx == app.processing_active_dataset_idx)
        if clicked:
            app.set_active_processing_dataset(idx)
        meta = ds.time_column or "<time not set>"
        imgui.text_disabled(f"  time: {meta}")
    imgui.end_child()

    imgui.same_line()

    # Active dataset / variable panel
    ds = app.active_processing_dataset
    imgui.begin_child("##processing_active_dataset", width=mid_w, height=panel_h, border=True)
    imgui.text("Active Dataset")
    imgui.separator()
    imgui.text_wrapped(ds.name)
    imgui.text(f"Rows: {len(ds.df)}")
    imgui.text(f"Columns: {len(ds.df.columns)}")
    imgui.spacing()

    time_candidates = candidate_time_columns(ds.df)
    if ds.time_column not in time_candidates and ds.time_column is not None:
        time_candidates.insert(0, ds.time_column)
    current_time = ds.time_column if ds.time_column else (time_candidates[0] if time_candidates else "<none>")
    if imgui.begin_combo("Time Feature", current_time):
        for col in time_candidates:
            selected = ds.time_column == col
            clicked, _ = imgui.selectable(col, selected)
            if clicked:
                try:
                    app.set_processing_time_metadata(app.processing_active_dataset_idx, time_column=col)
                except Exception as exc:
                    app.set_status(f"Time assignment failed: {exc}")
            if selected:
                imgui.set_item_default_focus()
        imgui.end_combo()

    current_tz = ds.timezone if ds.timezone in app.processing_timezones else app.processing_timezones[0]
    if imgui.begin_combo("Timezone", current_tz):
        for tz in app.processing_timezones:
            selected = ds.timezone == tz
            clicked, _ = imgui.selectable(tz, selected)
            if clicked:
                try:
                    app.set_processing_time_metadata(app.processing_active_dataset_idx, timezone=tz)
                except Exception as exc:
                    app.set_status(f"Timezone update failed: {exc}")
            if selected:
                imgui.set_item_default_focus()
        imgui.end_combo()

    current_unit = ds.posix_unit if ds.posix_unit in app.processing_posix_units else "ms"
    if imgui.begin_combo("POSIX Unit", current_unit):
        for unit in app.processing_posix_units:
            selected = ds.posix_unit == unit
            clicked, _ = imgui.selectable(unit, selected)
            if clicked:
                try:
                    app.set_processing_time_metadata(app.processing_active_dataset_idx, posix_unit=unit)
                except Exception as exc:
                    app.set_status(f"POSIX unit update failed: {exc}")
            if selected:
                imgui.set_item_default_focus()
        imgui.end_combo()

    if ds.time_column and POSIX_TIME_COL in ds.df.columns:
        imgui.text_colored("POSIX time ready", 0.6, 1.0, 0.6)
    else:
        imgui.text_colored("POSIX time not ready", 1.0, 0.6, 0.4)

    imgui.spacing()
    imgui.separator()
    imgui.text("Variables")
    imgui.begin_child("##processing_vars", width=0, height=0, border=True)
    for col in app.dataset_columns(ds):
        selected = col in app.processing_selected_variables
        clicked, _ = imgui.selectable(col, selected)
        if clicked:
            app.toggle_processing_variable(col)
    imgui.end_child()
    imgui.end_child()

    imgui.same_line()

    # Operations panel
    imgui.begin_child("##processing_ops", width=right_w, height=panel_h, border=True)
    imgui.text("Operations")
    imgui.separator()
    var_count = len(app.selected_processing_variables)
    dataset_count = len(app.selected_processing_datasets)
    imgui.text(f"Selected datasets: {dataset_count}")
    imgui.text(f"Selected variables: {var_count}")
    imgui.spacing()

    if var_count == 1:
        imgui.text("Single-variable operations")
        imgui.separator()
        if imgui.button("Take Derivative", width=180):
            app.processing_take_derivative()
        labels = [label for _, label in app.processing_quantify_methods]
        current_label = labels[app.processing_quantify_idx]
        if imgui.begin_combo("Quantify Variable", current_label):
            for idx, label in enumerate(labels):
                selected = idx == app.processing_quantify_idx
                clicked, _ = imgui.selectable(label, selected)
                if clicked:
                    app.processing_quantify_idx = idx
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        _, app.processing_window_size = imgui.slider_int("Window Size", int(app.processing_window_size), 1, 500)
        if imgui.button("Apply Quantification", width=180):
            app.processing_quantify_variable()
        imgui.spacing()

    if var_count >= 1:
        imgui.text("One-or-more-variable operations")
        imgui.separator()
        if imgui.button("Remove Variable(s)", width=180):
            app.processing_remove_variables()
        _, app.processing_poly_order = imgui.slider_int("Polynomial Order", int(app.processing_poly_order), 2, 6)
        if app.processing_poly_order >= 4:
            imgui.text_colored("High polynomial orders can explode the feature count quickly.", 1.0, 0.8, 0.4)
        if imgui.button("Polynomially Expand", width=180):
            app.processing_polynomial_expand()
        if imgui.button("Remove Duplicate Timestamps", width=220):
            app.processing_drop_duplicate_timestamps()
        if imgui.button("Drop Rows With NaNs", width=180):
            app.processing_drop_nan_rows()
        _, app.processing_interp_order = imgui.slider_int("Interpolation Order", int(app.processing_interp_order), 1, 5)
        if imgui.button("Interpolate Missing", width=180):
            app.processing_interpolate_missing()
        fill_dir = app.processing_fill_directions[app.processing_fill_direction_idx]
        if imgui.begin_combo("Fill Direction", fill_dir):
            for idx, direction in enumerate(app.processing_fill_directions):
                selected = idx == app.processing_fill_direction_idx
                clicked, _ = imgui.selectable(direction, selected)
                if clicked:
                    app.processing_fill_direction_idx = idx
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        if imgui.button("Fill Missing", width=180):
            app.processing_fill_missing()
        imgui.spacing()

    if var_count >= 2:
        imgui.text("Multi-variable operations")
        imgui.separator()
        _, app.processing_interaction_name = imgui.input_text("Interaction Name", app.processing_interaction_name, 128)
        if imgui.button("New Interaction Term", width=180):
            app.processing_new_interaction_term()
        _, app.processing_average_name = imgui.input_text("Average Name", app.processing_average_name, 128)
        if imgui.button("Merge Values (Average)", width=180):
            app.processing_merge_values_average()
        imgui.spacing()

    if dataset_count >= 1:
        imgui.text("Entire-set operations")
        imgui.separator()
        current_ds_method = app.processing_downsample_methods[app.processing_downsample_idx]
        if imgui.begin_combo("Downsample Method", current_ds_method):
            for idx, method in enumerate(app.processing_downsample_methods):
                selected = idx == app.processing_downsample_idx
                clicked, _ = imgui.selectable(method, selected)
                if clicked:
                    app.processing_downsample_idx = idx
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        _, app.processing_downsample_window = imgui.slider_int("Downsample Window", int(app.processing_downsample_window), 1, 200)
        ref_label = "<no reference>"
        if 0 <= app.processing_reference_dataset_idx < len(app.processing_datasets):
            ref_label = app.processing_datasets[app.processing_reference_dataset_idx].name
        if imgui.begin_combo("Reference Dataset", ref_label):
            clicked, _ = imgui.selectable("<no reference>", app.processing_reference_dataset_idx == -1)
            if clicked:
                app.processing_reference_dataset_idx = -1
            for idx, ref_ds in enumerate(app.processing_datasets):
                selected = idx == app.processing_reference_dataset_idx
                clicked, _ = imgui.selectable(ref_ds.name, selected)
                if clicked:
                    app.processing_reference_dataset_idx = idx
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        if imgui.button("Downsample Selected Set(s)", width=220):
            app.processing_downsample_selected_sets()
        imgui.spacing()

    if dataset_count >= 2:
        imgui.text("Multi-set operations")
        imgui.separator()
        _, app.processing_merged_dataset_name = imgui.input_text("Merged Set Name", app.processing_merged_dataset_name, 128)
        if imgui.button("Merge Sets By POSIX Time", width=220):
            app.processing_merge_selected_sets()
        imgui.text_wrapped("This merge trims each selected dataset to the overlapping time window first, then aligns rows by nearest POSIX timestamps.")

    imgui.end_child()

    imgui.spacing()
    imgui.separator()
    if imgui.button("Show Descriptive Statistics", width=220):
        app.show_processing_stats()
    imgui.same_line()
    if imgui.button("Save Active Dataset as CSV", width=220):
        app.save_active_processing_dataset()
    imgui.same_line()
    can_send = len(app.processing_datasets) == 1
    if not can_send:
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
    clicked_send = imgui.button("Send to Modeling Suite", width=220)
    if not can_send:
        imgui.pop_style_var()
    if clicked_send and can_send:
        app.send_active_dataset_to_modeling()
    if not can_send and imgui.is_item_hovered():
        imgui.set_tooltip("Only a single dataset can be sent to the modeling suite. Merge first if you currently have multiple datasets.")

    imgui.same_line()
    if imgui.button("Clear All Data", width=160):
        app.clear_all_processing_data()
    imgui.same_line()
    if imgui.button("Exit Processing Suite", width=180):
        app.exit_processing_window()

    imgui.end()

    if app.processing_stats_popup:
        vp = imgui.get_main_viewport()
        vx, vy = vp.pos
        vw, vh = vp.size
        win_w = max(700.0, vw * 0.82)
        win_h = max(420.0, vh * 0.78)
        win_x = vx + (vw - win_w) * 0.5
        win_y = vy + (vh - win_h) * 0.5
        imgui.set_next_window_position(win_x, win_y)
        imgui.set_next_window_size(win_w, win_h)
        flags = (
            imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_NO_SAVED_SETTINGS
        )
        imgui.begin("Descriptive Statistics", flags=flags)
        imgui.text_wrapped(f"Active dataset: {app.active_processing_dataset.name}")
        imgui.separator()
        avail_w, avail_h = imgui.get_content_region_available()
        child_h = max(120.0, avail_h - 48.0)
        imgui.begin_child("##stats_scroll", width=0, height=child_h, border=True)
        imgui.text_unformatted(app.processing_stats_text)
        imgui.end_child()
        if imgui.button("Close", width=140):
            app.processing_stats_popup = False
        imgui.end()


def draw_csv_headers_window(app: "SimImGuiApp") -> None:
    if not app.show_csv_headers_window:
        return

    _begin_fullscreen_window("##CSVColumnsFullscreen")
    imgui.text("CSV Column Assignment / Modeling Suite")
    if app.csv_path:
        imgui.same_line()
        imgui.text_disabled(f"({os.path.basename(app.csv_path)})")
    imgui.separator()
    imgui.spacing()

    headers = app.csv_headers
    if not headers:
        imgui.text("No headers loaded.")
        if imgui.button("Exit", width=140):
            app.show_csv_headers_window = False
        imgui.end()
        return

    if len(app.feature_mask) != len(headers):
        app.feature_mask = [False] * len(headers)
    if app.new_target_index >= len(headers):
        app.new_target_index = -1

    avail_w, avail_h = imgui.get_content_region_available()
    gap = 12
    panel_w = int((avail_w - gap) * 0.5)
    panel_h = int(avail_h - 120)

    imgui.begin_child("##features_panel", width=panel_w, height=panel_h, border=True)
    imgui.text("Features")
    imgui.separator()
    imgui.begin_child("##features_list", width=0, height=0, border=True)
    for idx, name in enumerate(headers):
        disabled = idx == app.new_target_index
        if disabled:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        changed, value = imgui.checkbox(f"{name}##feat_{idx}", app.feature_mask[idx])
        if changed and not disabled:
            app.feature_mask[idx] = value
        if disabled:
            imgui.pop_style_var()
            if imgui.is_item_hovered():
                imgui.set_tooltip("This column is currently selected as the target.")
    imgui.end_child()
    imgui.end_child()

    imgui.same_line(spacing=gap)
    app.draw_model_config_panel(width=panel_w, height=panel_h)

    imgui.spacing()
    imgui.separator()

    feat_count = sum(1 for on in app.feature_mask if on)
    tgt_name = headers[app.new_target_index] if app.new_target_index >= 0 else "<none>"
    imgui.text(f"Selected features: {feat_count}")
    imgui.same_line()
    imgui.text(f"Selected target: {tgt_name}")

    imgui.spacing()
    if imgui.button("Clear Features", width=160):
        app.feature_mask = [False] * len(headers)
    imgui.same_line()
    if imgui.button("Clear Target", width=160):
        app.new_target_index = -1
        app.target_index = -1
    imgui.same_line()
    if imgui.button("Exit Modeling Suite", width=200):
        app.show_csv_headers_window = False

    imgui.end()
