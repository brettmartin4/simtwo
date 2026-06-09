# Simulator 2 (Simtwo) [^simtwo]

### Suite of tools for quantum channel modeling with standalone GUI and backend support for SeQUeNCe experiment integration

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

# Demos

### Time-of-Flight Comparison

The GUI can be used in standalone mode to demonstrate photon travel time in relation to ground truth data, if available. Simply load a dataset, build a model within the modeling suite, and send the model and data to the observer view for observation. The code used to demonstrate this experiment programmatically in our paper is located in the examples directory, [here](https://github.com/brettmartin4/simtwo/blob/main/examples/demo_sequence_time_of_flight.py).

### Two-Node Entanglement Distribution Experiment (SeQUeNCe Comparison)

To verify how using a dynamic channel module impacts the performance of quantum networking experiments in SeQUeNCe, the [two-node entanglement distribution experiment](https://github.com/brettmartin4/simtwo/blob/main/examples/two_node_eg.ipynb) from the SeQUeNCe quantum network simulator is used as the baseline. Performance of the original experiment is compared against results from [our updated experiment notebook](https://github.com/brettmartin4/simtwo/blob/main/examples/simtwo_two_node_eg.ipynb) that contains experiment results using the Simtwo default single-factor physical model. The comparison in results between both experiments is plotted [here](https://github.com/brettmartin4/simtwo/blob/main/examples/two_node_eg_plotting.ipynb).

# References

Martin, B., Hodson, D., Wagner, T., Grimaila, M., Richards, A. M. (2026) "Simulator 2: A GUI-Enabled Suite of Tools for Dynamic Quantum Channel Modeling and Simulator Backend Integration." Pending publication (submitted to SoftwareX).

# To-do List:

 - (0) Validate modeling code with another notebook + load
 - (1) Go back to Simtwo 2NED experiment and add results for Simtwo experiment with delay model disabled (to verify that it matches the regular SeQUeNCe experiment results.
 - (2) Finish applying formatting changes and customization tools to observer mode
    - The customization panel should allow a plot to show up beneath the main observer mode plot that shows a preview of what the matplotlib-produced figure will look like
       - Parameters for this figure should be customizable from this panel, as well (font sizes, axis/header titles, etc)
    - Allow for selection of y axis value (or values if dual-axis) based on available models (time sync, prop delay, etc)
    - Update Poincare sphere plot to reflect stochastic drift model unless specified by some dropdown
    - Allow for multiple loaded models (with one being the active model) from which the plots can use to make visualizations
       - Change plot axis selection so either y axis can plot: a variable from the loaded dataset, inference from a loaded model, predictions from a physical model
 - (2.5) Create new Fig 2 for paper using Simtwo output
 - (3) For polarization physical model, use [von Mises-Fischer distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises_fisher.html) when sampling random walk and a modified version of [Qutip's Block sphere](https://qutip.org/docs/4.7/guide/guide-bloch.html) for the Poincare sphere
 - (4) Consider adding 1D [Perlin noise](https://dl.acm.org/doi/10.1145/566654.566636) option for jitter modeling (current model only samples a Gaussian at a user-defined or feature-derived stdev and mean)
 - (5) Convert jupyter notebooks to [Marimo](https://marimo.io/)
 - (6) Generate documentation
    - Look into [pdoc](https://pdoc.dev/) for generating code documentation
       - ~~Go back through and implement argument and return type annotations using the `from __future__ import annotations` standard from [Python PEP 563](https://peps.python.org/pep-0563/) (This will help with auto-generating docs later on, iirc)~~
 - ~~Migrate code from private repo here~~
 - ~~Finish and load test notebook for data processing suite~~
 - ~~(1) Complete modeling suite functionality~~
    - ~~Modify existing code so that target selection requires a "target type" to be selected (Time sync error, prop delay, path delay, polarization fidelity, Stokes vector, etc)~~
 - ~~Add filedialpy to toml~~
 - ~~Write example script for time-of-flight AND entanglement distribution SeQUeNCe experiment~~

[^simtwo]: <sub>Simtwo is short for “Simulator 2.” There is, at present, no Simulator 1. This is intentional. Work on Simfour will begin upon completion of this project.</sub>
