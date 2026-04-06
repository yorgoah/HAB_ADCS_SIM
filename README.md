# HAB_ADCS_SIM

Physics simulation of a high-altitude balloon payload's attitude dynamics. The project includes simulated reaction wheel control, momentum dumping with a non-interfering low-torque motor, and modeled sensors, actuators, and wind disturbances gathered from flight data.

## Simulation demo

![HAB ADCS simulation demo](docs/media/payload_animation_10s.gif)

## Installation and running

Make sure you have [Poetry installed](https://python-poetry.org/docs/#installation) before running this project.

From the repository root, run:

```bash
poetry install
poetry run python src/hab_adcs_sim/sim.py
