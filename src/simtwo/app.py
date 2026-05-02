import argparse

from simtwo.core.ui.main_window import run_app
from simtwo.core.backends.sequence_experiment_backend import SequenceBackend
from simtwo.core.backends.standalone_channel_backend import StandaloneBackend
from simtwo.core.SimulatorAdapter import SimulatorAdapter


def build_backend(mode: str):
    # TODO: can probably remove this later
    if mode == "sequence":
        raw_sim = SimulatorAdapter()
        return SequenceBackend(raw_sim)

    if mode == "standalone":
        return StandaloneBackend()

    raise ValueError(f"Unknown mode: {mode}")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sequence", "standalone"], default="standalone", help="Choose whether to run the full SeQUeNCe-backed experiment or the standalone channel workbench.")
    args = parser.parse_args()

    backend = build_backend(args.mode)
    run_app(backend)
