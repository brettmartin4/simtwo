"""
Docstring for examples.imgui_example. Add to these lkater before generating docs with whatever app I plan on using.

This is just the imgui implementation of the tkinter gui app
"""


import threading
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import csv
import os
import imgui
import glfw
from imgui.integrations.glfw import GlfwRenderer
from OpenGL import GL

import pynfd



# ui state (shared and safe from threads)
@dataclass
class UIState:
    lock: threading.Lock = field(default_factory=threading.Lock)

    # plot data
    epochs: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)

    # conditions display
    conditions: dict[str, Any] = field(default_factory=dict)

    # controls
    run_speed_ms: int = 100
    running: bool = False

    # messages / status
    status: str = ""

    def reset(self):
        with self.lock:
            self.epochs.clear()
            self.times.clear()
            self.conditions.clear()
            self.status = ""


# plot widget
# TODO: swap this out for matplotlib? If so, figure out way to have user-customizable text on images
def draw_line_plot(
    label: str,
    xs: list[int],
    ys: list[float],
    size: Tuple[float, float] = (700, 260),
    pad: float = 10.0,
):
    """
    draw line plot inside ImGui child region using win draw list
    """
    imgui.text(label)
    imgui.begin_child(f"##{label}_plot", width=size[0], height=size[1], border=True)

    draw_list = imgui.get_window_draw_list()
    x0, y0 = imgui.get_cursor_screen_pos()
    w, h = size

    # plot area
    left = x0 + pad
    right = x0 + w - pad
    top = y0 + pad
    bottom = y0 + h - pad

    # background + border
    draw_list.add_rect_filled(x0, y0, x0 + w, y0 + h, imgui.get_color_u32_rgba(0.10, 0.10, 0.10, 1.0))
    draw_list.add_rect(x0, y0, x0 + w, y0 + h, imgui.get_color_u32_rgba(0.60, 0.60, 0.60, 1.0))

    if len(xs) >= 2 and len(ys) >= 2:
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # avoid division by zero
        if xmax == xmin:
            xmax = xmin + 1
        if ymax == ymin:
            ymax = ymin + 1e-12

        def to_screen(x, y):
            sx = left + (x - xmin) / (xmax - xmin) * (right - left)
            sy = bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
            return sx, sy

        # axes lines
        draw_list.add_line(left, bottom, right, bottom, imgui.get_color_u32_rgba(0.50, 0.50, 0.50, 1.0), 1.0)
        draw_list.add_line(left, top, left, bottom, imgui.get_color_u32_rgba(0.50, 0.50, 0.50, 1.0), 1.0)

        # polyline
        color = imgui.get_color_u32_rgba(0.20, 0.70, 1.00, 1.0)
        thickness = 2.0
        for i in range(1, len(xs)):
            xA, yA = to_screen(xs[i - 1], ys[i - 1])
            xB, yB = to_screen(xs[i], ys[i])
            draw_list.add_line(xA, yA, xB, yB, color, thickness)

        # annotate minmax
        imgui.set_cursor_screen_pos((x0 + pad, y0 + h - pad - 18))
        imgui.text(f"x:[{xmin},{xmax}]   y:[{ymin:.3e},{ymax:.3e}]")
    else:
        imgui.set_cursor_screen_pos((x0 + pad, y0 + pad))
        imgui.text("No data yet...")

    # reserve
    imgui.dummy(w, h)
    imgui.end_child()


class SimImGuiApp:
    """
    ImGui GUI wrapper for simulator2.

    simulator reqs (same as the old Tk version):
      - nodes dict with 'A' and 'B'
      - observations list
      - current_epoch
      - set_run_speed(ms)
      - run_sim(sender, receiver, cb_plot, cb_conditions, cb_poincare)
      - cleanup_after_ids()
      - load_file(path)
      - export_file(path)
    """

    def __init__(self, simulator):
        self.sim = simulator
        self.ui = UIState()
        self.ui.run_speed_ms = 100

        # window handle
        #   app.window = window
        self.window = None

        # about win
        self.want_about_popup = False
        self._ctrl_a_latched = False

        # csv picker win info
        self.show_import_picker = False
        self.import_dir = os.getcwd()
        self.import_selected: str | None = None
        self.want_import_csv = False
        self._ctrl_i_latched = False

        # csv header info
        self.show_csv_headers_window = False
        self.csv_headers: list[str] = []
        self.csv_path: str = ""
        self.feature_mask: list[bool] = []
        self.target_index: int = -1

        # model config state vars
        self.model_type_options = ["default model", "existing model", "new model"]
        self.model_type_idx = 0  # 0=default, 1=existing, 2=new

        # cur model state
        self.existing_model_path = ""
        self.show_existing_model_picker = False

        # new model state
        self.new_model_name = "my_model"
        self.new_epochs = 50
        self.new_lr = 1e-3

        # new model target
        self.new_target_index = -1   # -1 means none selected


        # data souirce
        self.data_label: str = "Default example data"

        # text fallback
        self.path_input = ""

    # callback defs
    def cb_plot(self, epoch: int, travel_time_s: float):
        with self.ui.lock:
            self.ui.epochs.append(epoch)
            self.ui.times.append(travel_time_s)

    def cb_conditions(self, conditions: dict[str, Any]):
        with self.ui.lock:
            self.ui.conditions = dict(conditions)

    def cb_poincare(self, state):
        # Ignored for now (use custom poincare code from polarization research data exploration based on bloch wsphere)
        pass

    # actions
    def start(self):
        self.sim.set_run_speed(self.ui.run_speed_ms)
        sender = self.sim.nodes["A"]
        receiver = self.sim.nodes["B"]
        self.ui.running = True
        self.sim.run_sim(sender, receiver, self.cb_plot, self.cb_conditions, self.cb_poincare)

    def restart(self):
        try:
            self.sim.cleanup_after_ids()
        except Exception:
            pass

        self.ui.reset()
        self.sim.current_epoch = 0
        self.start()

    def stop(self):
        self.ui.running = False
        try:
            self.sim.cleanup_after_ids()
        except Exception:
            pass

    # csv import defs
    def import_csv(self):
        # Open picker win
        self.show_import_picker = True
        self.import_selected = None

    def _read_csv_headers(self, path: str) -> list[str]:
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
        return [h.strip() for h in headers if h is not None]

    def _load_csv_and_open_headers(self, path: str):
        # load into sim
        self.sim.load_file(path)

        # headers read
        try:
            headers = self._read_csv_headers(path)
        except Exception as e:
            headers = []
            with self.ui.lock:
                self.ui.status = f"CSV read error: {e}"
        else:
            with self.ui.lock:
                self.ui.status = f"Loaded: {path}"

        self.csv_path = path
        self.csv_headers = headers

        # right panel data label
        base = os.path.basename(path)
        self.data_label = f"CSV: {base}"

        # init select state 
        self.feature_mask = [False] * len(headers)
        self.target_index = -1

        # show header select win?
        self.show_csv_headers_window = True

    def export_results(self):

        res = pynfd.save_file_dialog(filter_list=["txt"])
        if res and res[0] == pynfd.Result.OK:
            path = res[1]
            self.sim.export_file(path)
            with self.ui.lock:
                self.ui.status = f"Exported: {path}"

    # SHORTCUT DEFS
    def _shortcut_ctrl_a(self) -> bool:
        w = self.window
        if w is None:
            return False
        ctrl = (glfw.get_key(w, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
                or glfw.get_key(w, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS)
        a = glfw.get_key(w, glfw.KEY_A) == glfw.PRESS
        return ctrl and a

    def _shortcut_ctrl_i(self) -> bool:
        w = self.window
        if w is None:
            return False
        ctrl = (glfw.get_key(w, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
                or glfw.get_key(w, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS)
        i_key = glfw.get_key(w, glfw.KEY_I) == glfw.PRESS
        return ctrl and i_key
    
    # helper funcs
    def _model_type(self) -> str:
        return self.model_type_options[self.model_type_idx]

    # menu bar
    def draw_menu_bar(self):
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked, _ = imgui.menu_item("Import CSV", "Ctrl+I", False, True)
                if clicked:
                    self.want_import_csv = True

                clicked, _ = imgui.menu_item("Export Results", None, False, True)
                if clicked:
                    self.export_results()

                imgui.separator()

                clicked, _ = imgui.menu_item("Exit", None, False, True)
                if clicked:
                    self.stop()
                    glfw.set_window_should_close(glfw.get_current_context(), True)

                imgui.end_menu()

            if imgui.begin_menu("Help", True):
                clicked, _ = imgui.menu_item("About", "Ctrl+A", False, True)
                if clicked:
                    self.want_about_popup = True
                imgui.end_menu()

            imgui.end_main_menu_bar()

    # import picker
    def draw_import_csv_picker_window(self):
        if not getattr(self, "show_import_picker", False):
            return

        opened = True
        opened, _ = imgui.begin("Import CSV", opened, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        if not opened:
            self.show_import_picker = False
            imgui.end()
            return

        # make sure dir exists
        if not os.path.isdir(self.import_dir):
            self.import_dir = os.getcwd()

        imgui.text("Directory:")
        imgui.same_line()
        imgui.text(self.import_dir)

        imgui.separator()

        # up one directory
        if imgui.button("Up", width=80):
            parent = os.path.dirname(self.import_dir)
            if parent and os.path.isdir(parent):
                self.import_dir = parent
                self.import_selected = None

        imgui.same_line()
        if imgui.button("Refresh", width=80):
            pass

        imgui.spacing()

        # list folders and csv
        # this check might be unnecessary. remove later?
        try:
            entries = os.listdir(self.import_dir)
        except Exception as e:
            with self.ui.lock:
                self.ui.status = f"List dir error: {e}"
            entries = []

        folders = sorted([x for x in entries if os.path.isdir(os.path.join(self.import_dir, x))])
        csvs = sorted([x for x in entries if os.path.isfile(os.path.join(self.import_dir, x)) and x.lower().endswith(".csv")])

        imgui.begin_child("##filelist", width=520, height=300, border=True)

        for name in folders:
            clicked, _ = imgui.selectable(f"[DIR] {name}", False)
            if clicked:
                new_dir = os.path.join(self.import_dir, name)
                if os.path.isdir(new_dir):
                    self.import_dir = new_dir
                    self.import_selected = None

        for name in csvs:
            selected = (self.import_selected == name)
            clicked, _ = imgui.selectable(name, selected)
            if clicked:
                self.import_selected = name

            # double click reqd
            if selected and imgui.is_mouse_double_clicked(0):
                full_path = os.path.join(self.import_dir, name)
                self._load_csv_and_open_headers(full_path)
                self.show_import_picker = False
                imgui.end_child()
                imgui.end()
                return

        imgui.end_child()

        imgui.spacing()
        imgui.separator()

        can_load = self.import_selected is not None

        if not can_load:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        clicked_load = imgui.button("Load", width=120)
        if not can_load:
            imgui.pop_style_var()

        if clicked_load and can_load:
            full_path = os.path.join(self.import_dir, self.import_selected)
            self._load_csv_and_open_headers(full_path)
            self.show_import_picker = False

        if not can_load and imgui.is_item_hovered():
            imgui.set_tooltip("Select a CSV file first")

        imgui.same_line()
        if imgui.button("Cancel", width=120):
            self.show_import_picker = False
            self.import_selected = None

        imgui.end()



    # csv headers
    def draw_csv_headers_window(self):
        if not getattr(self, "show_csv_headers_window", False):
            return

        # fullscreen (mmaybe make scalable later?)
        vp = imgui.get_main_viewport()
        x, y = vp.pos
        w, h = vp.size

        # offset menu bar to avoid issues with drawing underneath of it
        menu_h = imgui.get_frame_height()  # approx height of main menu bar (make dynamic later?)
        x0, y0 = x, y + menu_h
        w0, h0 = w, max(1.0, h - menu_h)

        imgui.set_next_window_position(x0, y0)
        imgui.set_next_window_size(w0, h0)

        flags = (
            imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_NO_SCROLLBAR
            | imgui.WINDOW_NO_SAVED_SETTINGS
        )

        imgui.begin("##CSVColumnsFullscreen", flags=flags)

        # header
        imgui.text("CSV Column Assignment")
        if self.csv_path:
            imgui.same_line()
            imgui.text_disabled(f"({os.path.basename(self.csv_path)})")

        imgui.separator()
        imgui.spacing()

        n = len(self.csv_headers)
        if n == 0:
            imgui.text("No headers loaded.")
            imgui.spacing()
            if imgui.button("Close", width=140):
                self.show_csv_headers_window = False
            imgui.end()
            return

        # keep state arrays in sync
        if len(self.feature_mask) != n:
            self.feature_mask = [False] * n
        if not isinstance(self.target_index, int):
            self.target_index = -1

        # Two independent panels side by side without bleed
        avail_w, avail_h = imgui.get_content_region_available()
        gap = 12
        panel_w = int((avail_w - gap) * 0.5)
        # leave space for footer/buttons
        panel_h = int(avail_h - 120)

        # feature list
        imgui.begin_child("##features_panel", width=panel_w, height=panel_h, border=True)
        imgui.text("Features")
        imgui.separator()
        imgui.begin_child("##features_list", width=0, height=0, border=True)

        for i, name in enumerate(self.csv_headers):
            disabled = (i == getattr(self, "new_target_index", -1))
            if disabled:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

            changed, val = imgui.checkbox(f"{name}##feat_{i}", self.feature_mask[i])
            if changed and not disabled:
                self.feature_mask[i] = val

            if disabled:
                imgui.pop_style_var()
                if imgui.is_item_hovered():
                    imgui.set_tooltip("This column is currently selected as the target.")

        imgui.end_child()
        imgui.end_child()

        imgui.same_line(spacing=gap)

        # model config righthand
        self.draw_model_config_panel(width=panel_w, height=panel_h)

        # footer
        imgui.spacing()
        imgui.separator()

        feat_count = sum(1 for on in self.feature_mask if on)
        tgt_name = self.csv_headers[self.target_index] if self.target_index >= 0 else "<none>"
        imgui.text(f"Selected features: {feat_count}")
        imgui.same_line()
        imgui.text(f"Selected target: {tgt_name}")

        imgui.spacing()

        if imgui.button("Clear Features", width=160):
            self.feature_mask = [False] * n
            if self.target_index >= 0:
                self.feature_mask[self.target_index] = False

        imgui.same_line()
        if imgui.button("Clear Target", width=160):
            self.target_index = -1

        imgui.same_line()
        if imgui.button("Close", width=160):
            self.show_csv_headers_window = False

        imgui.end()


    def draw_model_config_panel(self, width: int, height: int):
        imgui.begin_child("##model_config_panel", width=width, height=height, border=True)

        imgui.text("Model Configuration")
        imgui.separator()

        # dropdown
        current = self.model_type_options[self.model_type_idx]
        if imgui.begin_combo("Model Type", current):
            for i, label in enumerate(self.model_type_options):
                selected = (i == self.model_type_idx)
                clicked, _ = imgui.selectable(label, selected)
                if clicked:
                    self.model_type_idx = i
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # configurable stuff
        mode = self._model_type()

        if mode == "default model":
            # updated from carriage return (this is the proper method per the docs)
            imgui.text_wrapped(
                "Uses the simulator's default model/config. "
                "No additional settings required."
            )
            imgui.spacing()
            if imgui.button("Apply Default", width=160):
                # change hook later
                with self.ui.lock:
                    self.ui.status = "Default model selected."

        elif mode == "existing model":
            imgui.text("Load a previously trained model.")
            imgui.spacing()

            # Show currently selected path
            imgui.text("Model Path:")
            imgui.begin_child("##model_path_box", width=0, height=60, border=True)
            imgui.text_wrapped(self.existing_model_path if self.existing_model_path else "<none selected>")
            imgui.end_child()

            imgui.spacing()

            if imgui.button("Choose Model File...", width=220):
                # use picker already set
                self.show_existing_model_picker = True

            imgui.same_line()
            if imgui.button("Clear", width=100):
                self.existing_model_path = ""

            imgui.spacing()
            if imgui.button("Load Model", width=160):
                if not self.existing_model_path:
                    with self.ui.lock:
                        self.ui.status = "Pick a model file first."
                else:
                    # sim/model loading
                    with self.ui.lock:
                        self.ui.status = f"Loaded model: {os.path.basename(self.existing_model_path)}"

            # picker win
            if getattr(self, "show_existing_model_picker", False):
                self.draw_existing_model_picker_window()

        elif mode == "new model":
            imgui.text("Create/train a new model.")
            imgui.spacing()

            changed, self.new_model_name = imgui.input_text("Name", self.new_model_name, 128)
            changed, self.new_epochs = imgui.slider_int("Epochs", int(self.new_epochs), 1, 500)
            changed, self.new_lr = imgui.input_float("Learning Rate", float(self.new_lr), 0.0, 0.0, format="%.6f")

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # target selector
            headers = list(getattr(self, "csv_headers", []))
            if headers:
                # clamp index if needed
                if not isinstance(getattr(self, "new_target_index", -1), int):
                    self.new_target_index = -1
                if self.new_target_index >= len(headers):
                    self.new_target_index = -1

                current_label = headers[self.new_target_index] if self.new_target_index >= 0 else "<select target>"
                if imgui.begin_combo("Target Variable", current_label):
                    # "None" option
                    clicked, _ = imgui.selectable("<none>", self.new_target_index == -1)
                    if clicked:
                        self.new_target_index = -1

                    for i, name in enumerate(headers):
                        selected = (i == self.new_target_index)
                        clicked, _ = imgui.selectable(name, selected)
                        if clicked:
                            self.new_target_index = i
                            # IMPORTANT: ensure target is not also a feature
                            if hasattr(self, "feature_mask") and 0 <= i < len(self.feature_mask):
                                self.feature_mask[i] = False
                        if selected:
                            imgui.set_item_default_focus()
                    imgui.end_combo()
            else:
                imgui.text_disabled("Load a CSV to select a target variable.")

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if imgui.button("Create/Train", width=160):
                tgt = headers[self.new_target_index] if headers and self.new_target_index >= 0 else "<none>"
                with self.ui.lock:
                    self.ui.status = (
                        f"New model requested: {self.new_model_name} "
                        f"(epochs={self.new_epochs}, lr={self.new_lr:.6f}, target={tgt})"
                    )


        imgui.end_child()


    # main ui panels
    def draw_left_panel(self):
        with self.ui.lock:
            xs = list(self.ui.epochs)
            ys = list(self.ui.times)
        draw_line_plot("Photon Travel Time (seconds)", xs, ys, size=(740, 300))

    def draw_right_panel(self):
        imgui.begin_child("##right_panel", width=360, height=0, border=True)

        imgui.text("Controls")
        imgui.separator()

        changed, speed = imgui.slider_int("Run speed (ms)", self.ui.run_speed_ms, 10, 1000)
        if changed:
            self.ui.run_speed_ms = int(speed)
            # TODO: this check might be pointless, too. Consider removing after checking
            try:
                self.sim.set_run_speed(self.ui.run_speed_ms)
            except Exception:
                pass

        if imgui.button("Start", width=160):
            self.start()
        imgui.same_line()
        if imgui.button("Restart", width=160):
            self.restart()

        if imgui.button("Stop", width=330):
            self.stop()

        imgui.spacing()
        imgui.separator()

        imgui.text("Conditions")
        imgui.begin_child("##conditions", width=0, height=160, border=True)
        with self.ui.lock:
            conds = dict(self.ui.conditions)
        if conds:
            for k, v in conds.items():
                imgui.text(f"{k}: {v}")
        else:
            imgui.text("No conditions yet.")
        imgui.end_child()

        imgui.spacing()
        imgui.separator()

        # Data source display (as opposed tomanual file path section)
        imgui.text("Data Source")
        imgui.begin_child("##datasource", width=0, height=60, border=True)
        imgui.text_wrapped(self.data_label)
        imgui.end_child()

        with self.ui.lock:
            status = self.ui.status
        if status:
            imgui.spacing()
            imgui.text_colored(status, 0.9, 0.9, 0.2)

        imgui.end_child()

    # top level ui
    def draw(self):
        # Menu
        self.draw_menu_bar()

        # Ctrl+A opens About (latched)
        if self._shortcut_ctrl_a():
            if not self._ctrl_a_latched:
                self.want_about_popup = True
                self._ctrl_a_latched = True
        else:
            self._ctrl_a_latched = False

        if self.want_about_popup:
            imgui.open_popup("About")
            self.want_about_popup = False

        opened, _ = imgui.begin_popup_modal("About", True)
        if opened:
            imgui.text("simtwo - Quantum Simulation GUI")
            imgui.separator()
            imgui.text("Built with pyimgui + GLFW")
            imgui.text("Python 3.12")
            imgui.spacing()
            if imgui.button("Close"):
                imgui.close_current_popup()
            imgui.end_popup()

        # Ctrl+I opens Import picker (latched)
        if self._shortcut_ctrl_i():
            if not self._ctrl_i_latched:
                self.want_import_csv = True
                self._ctrl_i_latched = True
        else:
            self._ctrl_i_latched = False

        if self.want_import_csv:
            self.want_import_csv = False
            self.import_csv()

        # Secondary windows
        self.draw_import_csv_picker_window()
        self.draw_csv_headers_window()

        # Main layout window
        imgui.set_next_window_position(10, 30)
        imgui.set_next_window_size(1120, 520)
        imgui.begin("Simulation", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)

        imgui.columns(2, "main_cols", border=True)
        self.draw_left_panel()
        imgui.next_column()
        self.draw_right_panel()
        imgui.columns(1)

        imgui.end()



def run_app(simulator):
    # GLFW init
    # This is probably pointless too:
    #if not glfw.init():
    #    raise RuntimeError("Could not init GLFW")
    glfw.init()

    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    window = glfw.create_window(1150, 600, "simtwo - ImGui GUI", None, None)
    #if not window:
    #    glfw.terminate()
    #    raise RuntimeError("Could not create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    imgui.create_context()
    impl = GlfwRenderer(window)

    app = SimImGuiApp(simulator)
    app.window = window
    app._ctrl_a_latched = False
    app.want_about_popup = False

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()
        app.draw()
        imgui.render()

        GL.glClearColor(0.08, 0.08, 0.08, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    # cleanup
    app.stop()
    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    # TODO:
    # Replace this import with actual simulator adapter (check tk version for refd)
    
    # Ex:
    # from simtwo.core.simulator_gui_adapter import SimulatorAdapter
    # sim = SimulatorAdapter(...args here...)
    
    # For now, just use SimulatorAdapter
    from simtwo.core.SimulatorAdapter import SimulatorAdapter

    sim = SimulatorAdapter()
    run_app(sim)
