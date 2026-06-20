import argparse
import importlib.util
from pathlib import Path
import sys

from simtwo.core.ui.main_window import run_app
from simtwo.core.backends.sequence_experiment_backend import SequenceBackend
from simtwo.core.backends.standalone_channel_backend import StandaloneBackend
from simtwo.core.SimulatorAdapter import SimulatorAdapter

from simtwo.core.backends.gui_backend import build_sequence_gui_backend, build_standalone_gui_backend


def load_experiment_from_script(script_path: str):
    """Loads a backend experiment python file to interact with the GUI."""

    path = Path(script_path).resolve()
    module_name = "simtwo_user_experiment"
    spec = importlib.util.spec_from_file_location(module_name, path)

    # TODO: error check for file existence here?
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Can't load experiment script: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module # register first to avoid error
    spec.loader.exec_module(module)

    # Add updated path for new sructure:
    if hasattr(module, "build_plugin"):
        return "plugin", module.build_plugin()
    if hasattr(module, "PLUGIN"):
        return "plugin", module.PLUGIN

    # new returns include strings for type (tentatively calling old setup legacy for easy remembering)
    if hasattr(module, "build_sim"):
        return "legacy_sim", module.build_sim()
    if hasattr(module, "SIM"):
        return "legacy_sim", module.SIM

    raise RuntimeError("Experiment script must define either build_sim() func or SIM obj or build_plugin() or PLUGIN")


def build_backend(mode: str, experiment: str | None = None):
    """Builds the backend."""

    if mode == "sequence":
        if experiment:
            kind, raw_sim = load_experiment_from_script(experiment)

            if kind == "plugin":
                return build_sequence_gui_backend()
            elif kind == "legacy_sim":
                return SequenceBackend(raw_sim)

        else:
            raw_sim = SimulatorAdapter()
            return SequenceBackend(raw_sim)

    elif mode == "standalone":
        #return StandaloneBackend()
        return build_standalone_gui_backend()

    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    """Runs command-line entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sequence", "standalone"], default="standalone", help="Choose whether to run the full SeQUeNCe-backed experiment or the standalone channel workbench.")
    parser.add_argument("--experiment", default="", help="Optional path to custom SeQUeNCe backend experiment script.")
    args = parser.parse_args()

    backend = build_backend(args.mode, args.experiment or None)
    run_app(backend)


if __name__ == "__main__":
    main()