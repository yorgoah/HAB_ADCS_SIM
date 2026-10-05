import pickle
import numpy as np
import pandas as pd
import click
from sim_tools.integrator import ModelIntegrator, wrap_angle
from pathlib import Path
import json
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMETERS = ROOT / "config" / "sim_params" / "parameters_ec60.json"
RESULTS_DIR = ROOT / "results"

CHANNELS = (
    "time", "yaw", "ang_vel", "rw_i", "lt_torque", "rw_vel",
    "x", "y", "disturbance", "rw_torque", "error", "momentum", "pointing_state",
    "altitude",
)


def save_results(output_path: Path, df: pd.DataFrame, params: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({"parameters": params, "data": df}, f)


def _step_count(dt: float, total_time: float) -> int:
    # Same accumulation as the run loop, int(total_time / dt) + 1 can be off by one
    n = 0
    t = 0.0
    while t <= total_time:
        n += 1
        t += dt
    return n


def run_simulation(params: dict, seed: int | None = None, wind_seed: int | None = None) -> pd.DataFrame:
    """Run the model for the configured duration and log every channel.

    seed fixes the sensor noise and the wind, wind_seed overrides the wind only.
    """
    dt = params['simulation']['time_step']
    total_time = params['simulation']['duration']
    init_state = params['simulation']['initial_state']
    model = ModelIntegrator(dt, init_state, params, seed=seed, wind_seed=wind_seed)

    n = _step_count(dt, total_time)
    log = {name: np.empty(n, dtype=float) for name in CHANNELS}

    state = np.array(init_state, dtype=float)
    t = 0.0

    for i in range(n):
        log["time"][i] = t
        log["yaw"][i] = state[0]
        log["ang_vel"][i] = state[1]
        log["rw_i"][i] = state[2]
        log["lt_torque"][i] = state[3]
        log["rw_vel"][i] = state[4]
        log["x"][i] = state[5]
        log["y"][i] = state[6]
        log["disturbance"][i] = state[7]
        log["rw_torque"][i] = state[8]
        log["error"][i] = wrap_angle(state[0]-np.arctan2(state[6], state[5]))
        log["momentum"][i] = model.angular_momentum(state)
        log["altitude"][i] = model.disturbance.altitude(t)
        state = model.rk4_step(state, t)
        # State that set this step's command
        log["pointing_state"][i] = model.pointing.state.value
        t += dt

    return pd.DataFrame(log)


@click.command()
@click.option(
    "--parameters",
    default=DEFAULT_PARAMETERS,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
def simulate(parameters):
    with open(parameters, 'r') as f:
        params = json.load(f)

    df = run_simulation(params)
    output_path = RESULTS_DIR / f"simulation_results_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
    save_results(output_path, df, params)
    click.echo(f"Saved results to {output_path}.")

if __name__ == "__main__":
    simulate()
