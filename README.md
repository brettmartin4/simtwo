# Simulator 2 (Simtwo) [^simtwo]

### Suite of tools for quantum channel modeling with standalone GUI and backend support for SeQUeNCe experiment integration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fbrettmartin4%2Fsimtwo%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-2ea44f?logo=github)](https://brettmartin4.github.io/simtwo/)



Features configurable backend support for use with other simulators in the future. Make sure to check out the [Documentation!](https://brettmartin4.github.io/simtwo)

[Highlights](#highlights) · [Installation](#installation) · [Demos](#demos)· [Contribution](#contribution)· [References](#references) 


---

# Highlights

 - Single platform for data processing and modeling for timing and polarization errors
 - Backend interface for easy extension of quantum networking experiments
 - Validated on time-of-flight and two-node entanglement distribution in SeQUeNCe


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

---

# Demos

### Time-of-Flight Comparison

The GUI can be used in standalone mode to demonstrate photon travel time in relation to ground truth data, if available. Simply load a dataset, build a model within the modeling suite, and send the model and data to the observer view for observation. The code used to demonstrate this experiment programmatically in our paper is located in the examples directory, [here](https://github.com/brettmartin4/simtwo/blob/main/examples/demo_sequence_time_of_flight.py).

### Two-Node Entanglement Distribution Experiment (SeQUeNCe Comparison)

To verify how using a dynamic channel module impacts the performance of quantum networking experiments in SeQUeNCe, the [two-node entanglement distribution experiment](https://github.com/brettmartin4/simtwo/blob/main/examples/two_node_eg.ipynb) from the SeQUeNCe quantum network simulator is used as the baseline. Performance of the original experiment is compared against results from [our updated experiment notebook](https://github.com/brettmartin4/simtwo/blob/main/examples/simtwo_two_node_eg.ipynb) that contains experiment results using the Simtwo default single-factor physical model. The comparison in results between both experiments is plotted [here](https://github.com/brettmartin4/simtwo/blob/main/examples/two_node_eg_plotting.ipynb).

---

# Contribution

If you would like to contribute, please submit a pull request with a detailed summary of changes made.

---

# References

Martin, B., Hodson, D., Wagner, T., Grimaila, M., Richards, A. M. (2026) "Simulator 2: A GUI-Enabled Suite of Tools for Dynamic Quantum Channel Modeling and Simulator Backend Integration." Pending (submitted to SoftwareX for approval/review).


[^simtwo]: <sub>Simtwo is short for “Simulator 2.” There is, at present, no Simulator 1. This is intentional. Work on Simfour will begin upon completion of this project.</sub>
