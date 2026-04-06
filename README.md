# Background

The ALTAIR project aims to provide precise photometric calibration for ground-based telescopes by flying calibrated, and real-time optically measured, light sources on high-altitude balloons. Pointing of the light source into the telescope aperture is vital to reducing uncertainty related to attitude estimation. Since the emitted light's perceived brightness is a cosine function of its relative angle to the observer, a larger relative angle means that uncertainties in the relative angle estimate translate to large uncertainties in the measured brightness. The solution is to use a reaction wheel for fine azimuthal pointing of the payload and a momentum management system to prevent motor saturation, the design of this control system is supported by this simulation.

## Simulation

Physics simulation of a high-altitude balloon payload's attitude dynamics. The project includes simulated reaction wheel control, momentum dumping with a non-interfering low-torque motor, and modeled sensors, actuators, and wind disturbances gathered from flight data.

## Simulation demo

![HAB ADCS simulation demo](docs/media/payload_animation_10s.gif)

## Installation and running

Make sure you have [Poetry installed](https://python-poetry.org/docs/#installation) before running this project.

From the repository root, run:

```bash
poetry install
poetry run python src/hab_adcs_sim/sim.py
