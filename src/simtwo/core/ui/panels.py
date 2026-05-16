from __future__ import annotations

import os
from typing import TYPE_CHECKING

import imgui
from typing import TYPE_CHECKING

from simtwo.core.backends.protocol import ChannelModelConfig
from simtwo.core.modeling.model import SUPPORTED_MODEL_KINDS
from simtwo.core.ui.dialogs import draw_existing_model_picker_window
from simtwo.core.ui.plots import draw_line_plot

# caused circular import error when left out:
if TYPE_CHECKING:
    from simtwo.core.ui.main_window import SimImGuiApp



def draw_menu_bar(app: "SimImGuiApp") -> None:
    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File", True):
            clicked, _ = imgui.menu_item("Import CSV", "Ctrl+I", False, True)
            if clicked:
                app.want_import_csv = True

            clicked, _ = imgui.menu_item("Export Results", None, False, True)
            if clicked:
                app.export_results()

            imgui.separator()
            clicked, _ = imgui.menu_item("Exit", None, False, True)
            if clicked:
                app.stop()
                app.close_window()
            imgui.end_menu()

        if imgui.begin_menu("Help", True):
            clicked, _ = imgui.menu_item("About", "Ctrl+A", False, True)
            if clicked:
                app.want_about_popup = True
            imgui.end_menu()

        imgui.end_main_menu_bar()



def draw_left_panel(app: "SimImGuiApp") -> None:
    #with app.ui.lock:
    #    xs = list(app.ui.epochs)
    #    ys = list(app.ui.times)
    # draw_line_plot(app.plot_label, xs, ys, size=(740, 300))
    imgui.begin_child("##left_panel", width=0, height=0, border=False)

    imgui.text("Observer View")
    imgui.separator()

    plot_width = max(240.0, imgui.get_content_region_available_width() - 8.0)

    with app.ui.lock:
        xs = list(app.ui.epochs)
        ys = list(app.ui.times)

    if xs and ys:
        draw_line_plot(app.plot_label, xs, ys, size=(plot_width, 300))

    else:
        imgui.spacing()
        imgui.text_disabled("No model prediction data to display yet.")
        imgui.text_disabled("Press Start to begin generating predictions.")

    imgui.end_child()


def draw_right_panel(app: "SimImGuiApp") -> None:
    imgui.begin_child("##right_panel", width=360, height=0, border=True)

    imgui.text("Controls")
    imgui.separator()

    changed, speed = imgui.slider_int("Run speed (ms)", app.ui.run_speed_ms, 0, 100)
    if changed:
        app.ui.run_speed_ms = int(speed)
        app.backend.set_run_speed(app.ui.run_speed_ms)

    if imgui.button("Start", width=160):
        app.start()
    imgui.same_line()
    if imgui.button("Restart", width=160):
        app.restart()

    if imgui.button("Stop", width=330):
        app.stop()

    imgui.spacing()
    imgui.separator()

    imgui.text("Backend Mode")
    imgui.begin_child("##backendmode", width=0, height=45, border=True)
    imgui.text_wrapped(app.backend.get_mode_name())
    imgui.end_child()

    imgui.spacing()
    imgui.separator()

    imgui.text("Conditions")
    imgui.begin_child("##conditions", width=0, height=160, border=True)
    with app.ui.lock:
        conds = dict(app.ui.conditions)
    if conds:
        for key, value in conds.items():
            imgui.text(f"{key}: {value}")
    else:
        imgui.text("No conditions yet.")
    imgui.end_child()

    imgui.spacing()
    imgui.separator()

    imgui.text("Current Model")
    imgui.begin_child("##current_model", width=0, height=60, border=True)
    imgui.text_wrapped(app.current_model_name)
    imgui.end_child()

    imgui.spacing()
    imgui.separator()

    imgui.text("Data Source")
    imgui.begin_child("##datasource", width=0, height=60, border=True)
    imgui.text_wrapped(app.data_label)
    imgui.end_child()

    with app.ui.lock:
        status = app.ui.status
    if status:
        imgui.spacing()
        imgui.separator()
        imgui.text("Status")
        imgui.begin_child("##status_box", width=0, height=80, border=True)
        imgui.push_text_wrap_pos(0.0)
        imgui.push_style_color(imgui.COLOR_TEXT, 0.9, 0.9, 0.2, 1.0)
        imgui.text_unformatted(status)
        imgui.pop_style_color()
        imgui.pop_text_wrap_pos()
        imgui.end_child()

    imgui.end_child()



def draw_model_config_panel(app: "SimImGuiApp", width: int, height: int) -> None:
    imgui.begin_child("##model_config_panel", width=width, height=height, border=True)

    imgui.text("Model Configuration")
    imgui.separator()

    current = app.model_type_options[app.model_type_idx]
    if imgui.begin_combo("Model Type", current):
        for idx, label in enumerate(app.model_type_options):
            selected = idx == app.model_type_idx
            clicked, _ = imgui.selectable(label, selected)
            if clicked:
                app.model_type_idx = idx
            if selected:
                imgui.set_item_default_focus()
        imgui.end_combo()

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    mode = app._model_type()
    if mode == "default model":
        imgui.text_wrapped(
            "Uses the backend's default channel model/config. "
            "No additional settings are required."
        )
        imgui.spacing()
        if imgui.button("Apply Default", width=160):
            app.apply_model_config(
                ChannelModelConfig(
                    mode="default",
                    model_name="default_channel_model",
                    feature_names=app.selected_feature_names,
                    target_name=app.selected_target_name,
                )
            )

    elif mode == "existing model":
        imgui.text("Load a previously saved trained model bundle.")
        imgui.spacing()

        imgui.text("Model Path:")
        imgui.begin_child("##model_path_box", width=0, height=60, border=True)
        imgui.text_wrapped(app.existing_model_path if app.existing_model_path else "<none selected>")
        imgui.end_child()
        imgui.spacing()

        if imgui.button("Choose Model File...", width=220):
            app.show_existing_model_picker = True
        imgui.same_line()
        if imgui.button("Clear", width=100):
            app.existing_model_path = ""

        imgui.spacing()
        if imgui.button("Load Model", width=160):
            if not app.existing_model_path:
                app.set_status("Pick a model file first.")
            else:
                app.apply_model_config(
                    ChannelModelConfig(
                        mode="existing",
                        model_path=app.existing_model_path,
                        model_name=os.path.splitext(os.path.basename(app.existing_model_path))[0],
                        feature_names=app.selected_feature_names,
                        target_name=app.selected_target_name,
                    )
                )

        draw_existing_model_picker_window(app)

    elif mode == "new model":
        imgui.text("Train a sklearn model from the currently loaded CSV.")
        imgui.spacing()

        _, app.new_model_name = imgui.input_text("Name", app.new_model_name, 128)
        _, app.new_epochs = imgui.slider_int("Epochs", int(app.new_epochs), 1, 500)
        _, app.new_lr = imgui.input_float("Learning Rate", float(app.new_lr), 0.0, 0.0, format="%.6f")

        current_kind_label = app.model_kind_labels[app.model_kind_idx]
        if imgui.begin_combo("Training Model", current_kind_label):
            for idx, label in enumerate(app.model_kind_labels):
                selected = idx == app.model_kind_idx
                clicked, _ = imgui.selectable(label, selected)
                if clicked:
                    app.model_kind_idx = idx
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        if app.selected_model_kind == "random_forest":
            _, app.rf_n_estimators = imgui.slider_int("Trees", int(app.rf_n_estimators), 10, 1000)
            _, app.rf_max_depth = imgui.slider_int("Max Depth (0=None)", int(app.rf_max_depth), 0, 100)
        else:
            imgui.text_disabled("Linear regression has no extra hyperparameters in this UI yet.")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        headers = list(app.csv_headers)
        if headers:
            if app.new_target_index >= len(headers):
                app.new_target_index = -1
            current_label = headers[app.new_target_index] if app.new_target_index >= 0 else "<select target>"
            if imgui.begin_combo("Target Variable", current_label):
                clicked, _ = imgui.selectable("<none>", app.new_target_index == -1)
                if clicked:
                    app.new_target_index = -1
                    app.target_index = -1
                for idx, name in enumerate(headers):
                    selected = idx == app.new_target_index
                    clicked, _ = imgui.selectable(name, selected)
                    if clicked:
                        app.new_target_index = idx
                        app.target_index = idx
                        if idx < len(app.feature_mask):
                            app.feature_mask[idx] = False
                    if selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
        else:
            imgui.text_disabled("Load a CSV to select a target variable.")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text("Train / Validation / Test Split")
        changed, train_pct = imgui.slider_int("Train %", int(app.split_train_pct), 1, 98)
        if changed:
            app.split_train_pct = int(train_pct)
            app.split_validation_pct = min(int(app.split_validation_pct), max(1, 99 - int(app.split_train_pct)))
        max_val_pct = max(1, 99 - int(app.split_train_pct))
        changed, val_pct = imgui.slider_int("Validation %", int(app.split_validation_pct), 1, max_val_pct)
        if changed:
            app.split_validation_pct = int(val_pct)
        train_pct, val_frac, test_frac = app.current_split_fractions()
        imgui.text(f"Test %: {int(round(test_frac * 100))}")
        imgui.text_disabled("Splits are applied chronologically: train first, then validation, then test.")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if imgui.button("Train + Activate", width=160):
            app.train_and_activate_model()
        imgui.same_line()
        if imgui.button("Save Current Model", width=180):
            app.save_current_model()

        if app.last_training_summary:
            imgui.spacing()
            imgui.separator()
            imgui.text("Last Training Summary")
            imgui.begin_child("##training_summary", width=0, height=110, border=True)
            imgui.text_wrapped(
                f"Model: {app.last_training_summary.get('model_name', app.new_model_name)} "
                f"({SUPPORTED_MODEL_KINDS.get(app.last_training_summary.get('model_kind', ''), app.last_training_summary.get('model_kind', 'unknown'))})"
            )
            imgui.text(f"Target: {app.last_training_summary.get('target_name', '<unknown>')}")
            imgui.text(f"Samples: {app.last_training_summary.get('n_samples', 0)}")
            imgui.text(
                f"Split counts: train={app.last_training_summary.get('train_count', 0)}, "
                f"val={app.last_training_summary.get('validation_count', 0)}, "
                f"test={app.last_training_summary.get('test_count', 0)}"
            )
            imgui.text(f"Skipped rows: {app.last_training_summary.get('skipped_rows', 0)}")
            #imgui.text(f"Train RMSE: {app.last_training_summary.get('train_rmse', float('nan')):.6g}")
            #imgui.text(f"Train R^2: {app.last_training_summary.get('train_r2', float('nan')):.6g}")
            imgui.text(f"Train RMSE: {app.last_training_summary.get('train_rmse', float('nan')):.6g}")
            imgui.text(f"Validation RMSE: {app.last_training_summary.get('validation_rmse', float('nan')):.6g}")
            imgui.text(f"Test RMSE: {app.last_training_summary.get('test_rmse', float('nan')):.6g}")
            imgui.text(f"Train R^2: {app.last_training_summary.get('train_r2', float('nan')):.6g}")
            imgui.text(f"Validation R^2: {app.last_training_summary.get('validation_r2', float('nan')):.6g}")
            imgui.text(f"Test R^2: {app.last_training_summary.get('test_r2', float('nan')):.6g}")
            imgui.end_child()

    imgui.end_child()
