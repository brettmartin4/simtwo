# Simulator 2 (Simtwo)

Quantum channel simulator with standalone GUI and backend support for SeQUeNCe experiment integration.

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
