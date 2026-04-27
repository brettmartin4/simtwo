# Simulator 2 (Simtwo)

### Quantum channel simulator with standalone GUI and backend support for SeQUeNCe experiment integration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fbrettmartin4%2Fsimtwo%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)


Features configurable backend support for use with other simulators in the future.

<div align="center">
  [Installation](#installation) · [References](#references)
</div>

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
