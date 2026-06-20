"""
GUI expects a simulator object that provides these at the bare minimum:
    - nodes: dict with keys A and B
    - observations: a sequence so len(simulator.observations) works fine
    - current_epoch: int
    - run_sim(sender, receiver, update_plot, update_conditions, update_poincare_sphere)
    - set_run_speed(value: int)
    - cleanup_after_ids()
    - load_file(path: str)
    - export_file(path: str)
"""

# do not get rid of this (it will cause a runtime error when the example tries to grab the sim class)
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


# Bloch sphere for poincare rendering
from qutip import Qobj, Bloch



class SimulationGUI(ctk.CTk):
    """GUI for displaying simulator res. Sim needs to run in a background thread.
    All UI updates are marshalled onto the Tk main thread using self.after() def.
    """

    def __init__(self):
        """Initialize the SimulationGUI instance."""
        super().__init__()

        self.times: list[float] = []
        self.epochs: list[int] = []
        self.simulator: Any | None = None

        self._build_gui()

    # Entrypoint to sim
    def run_sim(self, simulator: Any):
        """Attach a simulator and enter Tk mainloop."""
        self.simulator = simulator
        self.mainloop()

    # building UI
    def _build_gui(self):
        """Builds gui window and contents."""
        self.title("Quantum Simulation Results")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("dark-blue")

        # Frame for all left side content (clock time plot)
        self.left_frame = ctk.CTkFrame(master=self)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Setup the plot for clock time
        self.figure = Figure(figsize=(8, 4))
        self.plot = self.figure.add_subplot(111)
        self._reset_plot_axes()
        self.figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figure, self.left_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(pady=12, padx=10, fill=tk.BOTH, expand=True)

        # Frame for all right side content (Bloch sphere and Conditions)
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.pack(pady=12, padx=10, side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._bloch_enabled = True

        self.bloch_figure = Figure(figsize=(2, 2))
        self.bloch_plot = self.bloch_figure.add_subplot(111, projection="3d")
        self.bloch = Bloch(axes=self.bloch_plot)

        self.bloch_canvas = FigureCanvasTkAgg(self.bloch_figure, self.right_frame)
        self.bloch_canvas_widget = self.bloch_canvas.get_tk_widget()
        self.bloch_canvas_widget.pack(pady=12, padx=10, side=tk.TOP, fill=tk.BOTH, expand=True)

        # Default state
        self.bloch.add_states(Qobj([1, 0]))
        self.bloch.render()
        self.bloch_canvas.draw()

        # Text box for environmental conditions
        self.conditions_text = tk.Text(self.right_frame, height=10, width=25)
        self.conditions_text.pack(pady=12, padx=10, side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Restart button
        self.restart_button = ctk.CTkButton(
            self.right_frame, text="Restart Simulation", command=self.restart_simulation
        )
        self.restart_button.pack(pady=12, padx=10, side=tk.BOTTOM)

        # Speed control
        self.variable = tk.IntVar(value=100)
        self.spinbox = tk.Spinbox(
            self.right_frame,
            from_=10,
            to=1000,
            increment=10,
            textvariable=self.variable,
            width=10,
            command=self.on_speed_change,
        )
        self.spinbox.pack(pady=10)

        # MENU
        self.menu = tk.Menu(self)
        self.config(menu=self.menu)

        filemenu = tk.Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Import CSV", command=self.load_file)
        filemenu.add_command(label="Export Results", command=self.export_file)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)

        helpmenu = tk.Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="About", command=self._about)

        # Start button (not in original, but handy)
        self.start_button = ctk.CTkButton(
            self.right_frame, text="Start Simulation", command=self.start_simulation
        )
        self.start_button.pack(pady=12, padx=10, side=tk.BOTTOM)

    def _reset_plot_axes(self):
        """Returns timing plot axes to default values."""
        self.plot.clear()
        self.plot.set_title("Photon Travel Time")
        self.plot.set_xlabel("Epoch")
        self.plot.set_ylabel("Travel Time (seconds)")
        self.plot.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.plot.yaxis.set_major_formatter(FormatStrFormatter("%.3e"))
        self.plot.set_xlim(0, 10)

    def _about(self):
        """Handles contents of the "about" section."""
        messagebox.showinfo(
            "About",
            "Simulation 2 (GUI Edition!)\n\nCurrently plots travel time vs epoch and shows per-epoch conditions with jitter and thermal influence.\n"
            "Bloch sphere functionality coming soon.",
        )

    # for handling window exits
    def on_close(self):
        """Handle the close event (window exit)."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            try:
                if self.simulator is not None:
                    self.simulator.cleanup_after_ids()
            except Exception as e:
                print(f"Error while stopping simulator: {e}")
            sys.exit()

    # UI controls
    def on_speed_change(self):
        """Handle the speed change event. TODO: Scheduled for removal since all plots are auto-generated in the most recent build."""
        if self.simulator is None:
            return
        try:
            self.simulator.set_run_speed(self.variable.get())
        except Exception as e:
            messagebox.showwarning("Speed", f"Could not set speed: {e}")

    def restart_simulation(self):
        """Handle restart simulation."""
        if self.simulator is None:
            return

        # Stop any running work first
        try:
            self.simulator.cleanup_after_ids()
        except Exception:
            pass

        self.times = []
        self.epochs = []
        self.simulator.current_epoch = 0

        self._reset_plot_axes()
        self.canvas.draw()

        try:
            self.simulator.set_run_speed(self.variable.get())
        except Exception:
            pass

        self.start_simulation()

    # File bar items
    def load_file(self):
        """Handles loading a file into the GUI."""
        if self.simulator is None:
            return
        file_path = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if file_path:
            try:
                self.simulator.load_file(file_path)
            except Exception as e:
                messagebox.showerror("Import", f"Failed to load file:\n{e}")
        else:
            messagebox.showinfo("No File", "No file was selected.")

    def export_file(self):
        """Handle exporting a file from the GUI."""
        if self.simulator is None:
            return
        file_path = filedialog.asksaveasfilename(
            title="Save as",
            defaultextension=".txt",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
        )
        if file_path:
            try:
                self.simulator.export_file(file_path)
            except Exception as e:
                messagebox.showerror("Export", f"Failed to export:\n{e}")
        else:
            messagebox.showinfo("Export Cancelled", "No file was selected for export.")

    # SIM START
    def start_simulation(self):
        """Begin simulation in background thread."""
        if self.simulator is None:
            messagebox.showwarning("Simulator", "No simulator attached.")
            return
        try:
            sender = self.simulator.nodes["A"]
            receiver = self.simulator.nodes["B"]
        except Exception as e:
            messagebox.showerror("Simulator", f"Simulator missing nodes A/B: {e}")
            return

        # Sim run in a background thread...
        # Tk HAS TO BE updated on main thread, so callbacks marshal updates via the after() def
        self.simulator.run_sim(sender, receiver, self.update_plot, self.update_conditions, self.update_poincare_sphere)

    # The following are callbacks for the UI (for updating values, etc-- minus pol functionality)
    def update_plot(self, epoch: int, travel_time: float):
        """Thread-safe wrapper for plot updates."""
        self.after(0, self._update_plot_ui, epoch, travel_time)

    def _update_plot_ui(self, epoch: int, travel_time: float):
        """Updates the UI for the plot."""
        self.times.append(travel_time)
        self.epochs.append(epoch)
        self.plot.plot(self.epochs, self.times, marker="o", color='black', linestyle="-")
        self.plot.yaxis.set_major_formatter(FormatStrFormatter("%.3e"))
        self.plot.xaxis.set_major_locator(MaxNLocator(integer=True))
        try:
            if self.simulator is not None:
                self.plot.set_xlim(0, max(1, len(self.simulator.observations) - 1))
        except Exception:
            pass
        self.canvas.draw()

    def update_conditions(self, conditions: dict[str, Any]):
        """A threadsafe wrapper for conditions textbox"""
        self.after(0, self._update_conditions_ui, conditions)

    def _update_conditions_ui(self, conditions: dict[str, Any]):
        self.conditions_text.delete("1.0", tk.END)
        # TODO: come back to this later and figure out another way to set this string so I can format floats properly
        conditions_display = "\n".join(f"{k}: {v}" for k, v in conditions.items())
        self.conditions_text.insert(tk.END, conditions_display)

    def update_poincare_sphere(self, state: Any):
        """callback: update Bloch sphere"""
        if not self._bloch_enabled:
            return
        # marshal to UI thread
        self.after(0, self._update_bloch_ui, state)

    def _update_bloch_ui(self, state: Any):
        try:
            self.bloch.clear()
            self.bloch.add_states(Qobj(state))
            self.bloch.render()
            self.bloch_canvas.draw()
        except Exception:
            # TODO: implement this later with the rstdev models
            # basically, just create prob dist around horizontal pol state for the mean.
            # Keep this non-fatal since im ignoring polarization for now
            pass
