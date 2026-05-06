# Simulator 2 (Simtwo) [^simtwo]

### Quantum channel simulator with standalone GUI and backend support for SeQUeNCe experiment integration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fbrettmartin4%2Fsimtwo%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)


Features configurable backend support for use with other simulators in the future.

[Installation](#installation) · [References](#references) · [To Do](#to-do-list)


---

# Installation

It is recommended to install Simulator 2 from within a virtual environment to avoid problems with mismatched dependency versions. The package can be installed directly from the root project directory (the directory containing this README) using the following command:

```bash
$ pip install . --no-build-isolation
```

To launch the standalone simulator GUI, either launch it from the command line with `python sim.py`, or import the `SimulatorGUI` class and call the `run` method:


```python
from simtwo.gui import SimulatorGUI

gui = SimulatorGUI()
gui.run()
```

# References

Lorem ipsum.

# To-do List:

 - ~~Migrate code from private repo here~~
 - Finish and load test notebook for data processing suite
 - Complete modeling suite functionality
    - Modify existing code so that target selection requires a "target type" to be selected (Time sync error, prop delay, path delay, polarization fidelity, Stokes vector, etc)
 - Validate modeling code with another notebook + load
 - Finish applying formatting changes and customization tools to observer mode
    - Allow for selection of y axis value (or values if dual-axis) based on available models (time sync, prop delay, etc)
    - Update Poincare sphere plot to reflect stochastic drift model unless specified by some dropdown
    - Allow for multiple loaded models (with one being the active model) from which the plots can use to make visualizations
       - Change plot axis selection so either y axis can plot: a variable from the loaded dataset, inference from a loaded model, predictions from a physical model
 - Write example script for time-of-flight or entanglement distribution SeQUeNCe experiment
 - Look into [pdoc](https://pdoc.dev/) for generating code documentation
    - Go back through and implement argument and return type annotations using the `from __future__ import annotations` standard from [Python PEP 563](https://peps.python.org/pep-0563/) (This will help with auto-generating docs later on, iirc)

---

[^simtwo]: <sub>Simtwo is short for “Simulator 2.” There is, at present, no Simulator 1. This is intentional. Work on Simfour is currently underway.</sub>
