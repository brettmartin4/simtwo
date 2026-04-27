---
title: 'Simulator 2: A GUI-Based Quantum Communication Network Channel Simulator with Support for SeQUeNCe'
tags:
  - Python
  - Quantum communication networks
  - Modeling and simulation
authors:
  - name: Brett M. Martin
    orcid: 0009-0001-7147-9923
    corresponding: true
    affiliation: 1
  - name: Douglas D. Hodson
    affiliation: 1
  - name: Torrey J. Wagner
    affiliation: 2
  - name: Michael R. Grimaila
    affiliation: 2
affiliations:
 - name: Department of Electrical and Computer Engineering, United States Air Force Institute of Technology, United States
   index: 1
   ror: 00hx57361
 - name: Department of Systems Engineering, United States Air Force Institute of Technology, United States
   index: 2
date: 20 April 2026
bibliography: paper.bib
---

# Summary

Quantum communication networks leverage the unique quantum-mechanical properties of photons to encode and distribute data over large geographic distances for tasks such as secure communications and distributed quantum computing. The fiber optic links that bridge the nodes on a quantum network, which are often deployed over the same telecommunications fiber as classical data streams, are susceptible to environmental noise that drive error in quantum channels in the form of time synchronication error, propagation delay, and polarization drift. While most modern simulators for quantum networks include provisions for these types of error, these implementations are overly simplified, with static quantum states, delays, and timing errors assumed over quantum channels during transmission. Because some quantum network protocols, such as entanglement distribution and Hong-Ou-Mandel measurements [@Hong87] require strict polarization compensation and picosecond-precision timing, the utility of these simulators for realistic, metropolitan-scale quantum network experiments is significantly inadequate.

We present Simulator 2, an open-source python package for quantum channel simulation for use in broader quantum communication network simulation frameworks. The package extends the Simulator for QUantum Network Communication (SeQUeNCe) [@Wu21; Kettimuthu19] with dynamic channel modules while also providing standalone support via an easy-to-use graphical user interface. As opposed to workflows employed in previous and ongoing studies that required hand-editing scripts across multiple stages of research, Simulator 2 offers a streamlined, multi-stage environment for data processing, modeling, channel configuration, and experiment output visualization.

Simulator 2 is geared towards quantum network researchers studying fiber-optic-based quantum communication networks in which the material properties of fused silica such as thermal expansion render them susceptible to various exogenous environmental stimuli. The package supports physical and stochastic models for propagation (round-trip) delay, hardware jitter, polarization drift, and time synchronization (clock) error. User-defined data-driven model support also exists through a modeling suite which allows for models to be trained on real-world data for use in simulation for error generation. The graphical user interface includes data-processing, modeling, and observer windows to aid in the configuration and inspection of channel effects without the need to alter the underlying experiment code.

One key feature of Simulator 2 is the backend abstraction functionality. By allowing for backend abstraction, the simulator can be used as either a standalone application for processing data and generating models, or as an extension layer for use in SeQUeNCe experiments. The result is an environment where researchers are able to maintain use of existing event-driven SeQUeNCe experiments while providing visualization of expected network conditions and channel behavior that provides a more accurate reflection of real-world quantum networking testbeds.

# Statement of need

Standing up a dedicated quantum communication network (QCN) is an extremely costly enterprise. As such, many of the quantum network testbeds under research and development today are deployed over commercially-installed telecommunications fiber links, using wavelength-division multiplexing (WDM) to transmit quantum data over distinct frequency bands over the same link as classical data--a feature referred to as coexistence [@Gerrits22]. Because of this, and by virtue of single-mode telecommunications fiber links often being installed aerially or near public transit infrastructure, these channels are susceptible to environmental noise from a multitude of sources from thermal loading to mechanical stress. This noise often manifests as polarization drift, photon loss, and time synchronization error. For a sufficiently robust QCN to function as intended, the development of error-corrective measures is critical.

Due to the high costs associated with standing up a quantum network and the limits on scalability, it is vitally important to be able to design and evaluate quantum networks within a simulation environment. This is particularly true of metropolitan-scale, fiber-based networks like the Washington D.C. Metropolitan Quantum Network Research COnsortium (DC-QNet) where channel behavior is influenced by environmental factors such as ambient temperature and mechanical stress. These effects manifest in the form of error that impacts synchronization between nodes, polarization stability, and photon travel times.

SeQUeNCe is especially useful because of the modular design interface, its open-source code base, and is used primarily as a protocol and system-level research platform. However, because SeQUeNCe assumes constant or simplified physical layer behaviors such as propagation delay, the dynamic nature of these behaviors is nost to abstraction. Many research questions require more dynamic channel behavior than what is currently provided in SeQUeNCe. For instance, researchers may be interested in evaluating how time synchronization error impacts critical networking protocols while preserving their SeQUeNCe experiment code.

Simulator 2 addresses this gap and specifically targets researchers who require a heightened level of quantum channel behavior realism in protocol-level simulation experiments. This is especially useful in contexts where experimental data for a particular testbed exists and researchers are interested in building testbed-specific error models as a substitute for the default, fixed error models included in SeQUeNCe.

# State of the field 

A number of quantum network simulators exist today that support this type of research with varying levels of abstraction. SeQUeNCe, developed at Argonne National Laboratory, is an open-source, discrete event simulator for higher-level protocol, entanglement generation, and network topology and routing research. Other simulators such as NetSquid [@Coopmans21] offer more comprehensive physical-layer modeling. QuISP [@Satoh22], another open-source simulator, focuses on more realistic, noisy conditions and respective simulated fault tolerance and error corrective measures one would expect in an operational QNet. The error models in QuISP, however, are abstract and do not explicitly account for environmental-driven sources of error. A recent survey by [@Bel25] provides a comprehensive analysis of the most widely-used quantum network simulators and their respective use-cases. Among them, those that do include meaningful representations of error model them as abstract, time-driven processes.

Simulator 2, rather than serving as a substitute for any of these platforms, serves as an extension of SeQUeNCe that adds various tools for dynamic channel modeling, model training pipelines, and experiment visualization such that it can be seamlessly integrated into pre-existing SeQUeNCe experiments. The extension-based design of Simulator 2 serves as the main build-versus-contribute justification for the package. The scholarly contribution is not solely an interface layer built on top of a pre-existing simulation framework. Rather, it combines several capabilities, some of which are available in standalone form, that would otherwise be unavailable to SeQUeNCe experiments:

1. Highly configurable physical and stochastic error models
2. Support for data-driven models for capturing real-world behaviors
3. A consolidated graphical user interface for data processing and modeling
4. Backend abstraction support for seamless SeQUeNCe experiment integration

# Software design

An explanation of the trade-offs you weighed, the design/architecture you chose, and why it matters for your research application. This should demonstrate meaningful design thinking beyond a superficial code structure description.

TODO: explain reason for design choice of backend abstraction. Also, explain why data processing and modeling pipelines were consolidated into a single UI (and maybe talk about how the observer view was added for a convenient means of verifying behavior of models created in the modeling suite). Lastly, go over a representative use-case (probably a simple time-of-flight or bell state distribution experiment) to just explain how this library works in tandem with SeQUeNCe for experiments. Probably shouldn't go into experiment specifics here, but at least show how the design is implemented such that it extends SeQUeNCe. Can create a much more detailed tutorial for the code writeup/documentation later, but not for this paper. 

# Research impact statement

Simulator 2 is currently being developed for current and ongoing research at both the United States Air Force Institute of Technology and the Laboratory of Telecommunications Science involving quantum channel behavior and error processes in fiber-based quantum communication networks. As an immediate impact, Simulator 2 serves as valuable research infrastructure. The application GUI alone provides a reusable pipeline for all aspects of quantum channel error and noise modeling for use in reproducible experiments in simulations. The application also provides a simple and easy alternative to manually constructing data processing and modeling pipelines and integrating them into simulators like SeQUeNCe.

Simulator 2 shows particular promise as a means of supporting reproducible experiments for comparative studies in both simulated and real-world quantum networks. The backend interface is especially useful for its ability to support high-level SeQUeNCe simulations over a broad range of different channel module configurations. This would allow for a more efficient means of comparing nominall, stochastic, physical, and data-driven channel behaviors under a standardized implementation framework. This provides excellent near-term significance as a software package as it enables rigorous research workflows as opposed to just one-off analyses.

# AI usage disclosure

AI was used to assist in the collection of literature related to this topic as a substitute for traditional search engine use. Papers deemed to be useful in background research efforts, particularly in the search for other related software, were used. Commit messages for Github repo were automatically generated by Copilot.

# References
