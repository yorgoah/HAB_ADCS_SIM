<h1 align="center">HAB ADCS Simulation</h1>

<p align="center"><i>Keeping a balloon payload pointed at a telescope from 20 km up.</i></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white" alt="Python 3.14">
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/status-flight%20tested-2ea44f" alt="Flight tested">
</p>

<p align="center">
  <img src="docs/figures/pointing_recovery.gif" alt="Simulated payload saturating its momentum wheel and recovering">
</p>

<p align="center"><sub>A gust pushes the momentum wheel to its speed limit. The flight computer spins it back down, steadies the payload and finds the ground station again.</sub></p>

## Overview

ALTAIR (Airborne Laser for Telescopic Atmospheric Interference Reduction) flies a calibrated light source on a high-altitude balloon, as an artificial star that ground-based telescopes can calibrate against. The catch is that the light has to face the telescope: the payload must hold its heading within ±8° for at least 30 s at a time, while hanging under a balloon that the wind keeps twisting.

This repo is the physics simulation I built to design that pointing system. It models the payload spinning in the wind, the momentum wheel that turns it, the pivot motor that keeps the wheel from saturating and the flight computer that runs them, so controllers can be tuned on the computer before they fly.

## Payload

<p align="center"><img src="docs/figures/flight_configuration.png" width="780" alt="ALTAIR balloon, payload CAD model and pivot motor cross-section"></p>

| Part | Job |
|---|---|
| Integrating sphere | The light source the telescope looks at |
| Servo | Tilts the sphere toward the telescope |
| Momentum wheel | Speeds up or slows down to turn the payload |
| Pivot motor | Pushes against the flight train so the wheel can shed its momentum |
| Carbon-fiber booms | Rigged to the balloon neck to stop the payload swinging |
| Pixhawk, magnetometer, dual GNSS | Work out which way the payload is facing |

## What is modelled

- Yaw dynamics of the payload, the momentum wheel and the pivot motor
- Brushless motor models with back-EMF speed limits and bearing friction
- Noisy gyro, IMU, tachometer and GNSS measurements
- Wind made from a real flight: every run gets a brand new gust history with the same character as the logged one, and both the gusts and the steady wind fade as the air thins on the way up
- The same pointing state machine as the flight computer
- Monte Carlo campaigns that throw many slightly different payloads and winds at the controllers

<details>
<summary><b>Momentum wheel saturation</b></summary>
<br>

The wheel can only spin so fast. When it hits its limit it can't turn the payload any further, so the flight computer drops the target, brings the wheel back to its middle speed (the payload spins as it takes up that momentum), damps the spin and starts pointing again. That's the GIF at the top. The pivot motor is there so this rarely has to happen.

<p align="center"><img src="docs/figures/pointing_state_machine.png" width="760" alt="Pointing state machine for momentum-wheel-only control"></p>

</details>

## Repository layout

```
src/sim_tools/        dynamics, motors, sensors, controller, pointing state machine, wind
src/hab_adcs_sim/     single runs, Monte Carlo campaigns, figures
config/sim_params/    parameter file
config/campaigns/     Monte Carlo campaign files
config/ressources/    flight data used for the wind
tests/                pytest suite
```

## About

Made by [Yorgo Abou Haidar](https://github.com/yorgoah) for ALTAIR, a McGill University and University of Victoria project.
