import argparse
import importlib.util
from pathlib import Path

from simtwo.core.ui.main_window import run_app
from simtwo.core.backends.sequence_experiment_backend import SequenceBackend
from simtwo.core.backends.standalone_channel_backend import StandaloneBackend
from simtwo.core.SimulatorAdapter import SimulatorAdapter


def load_experiment_from_script(script_path: str):

    path = Path(script_path).resolve()
    spec = importlib.util.spec_from_file_location("simtwo_user_experiment", path)

    # TODO: error check for file existence here?

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "build_sim"):
        return module.build_sim()

    if hasattr(module, "SIM"):
        return module.SIM

    raise RuntimeError("Experiment script must define either build_sim() func or SIM obj")


def build_backend(mode: str, experiment: str | None = None):

    if mode == "sequence":
        if experiment:
            raw_sim = load_experiment_from_script(experiment)
        else:
            raw_sim = SimulatorAdapter()
        return SequenceBackend(raw_sim)

    if mode == "standalone":
        return StandaloneBackend()

    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sequence", "standalone"], default="standalone", help="Choose whether to run the full SeQUeNCe-backed experiment or the standalone channel workbench.")
    parser.add_argument("--experiment", default="", help="Optional path to custom SeQUeNCe backend experiment script.")
    args = parser.parse_args()

    backend = build_backend(args.mode, args.experiment or None)
    run_app(backend)


# NOTE: The app worked without this previously, but broke when I implemented the sequence backend functionality. Not sure why.
if __name__ == "__main__":
    main()