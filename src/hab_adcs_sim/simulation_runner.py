import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import click
from sim_tools.integrator import ModelIntegrator, wrap_angle
from pathlib import Path
import json
import time

DEFAULT_PARAMETERS = Path(__file__).resolve().parents[2] / "config" / "sim_params" / "parameters_ec60.json"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "results" / f"simulation_results_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"


def _save_results(output_path: Path, df: pd.DataFrame, params: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({"parameters": params, "data": df}, f)


def _run_simulation(params: dict) -> pd.DataFrame:
    dt = params['simulation']['time_step']
    total_time = params['simulation']['duration']
    init_state = params['simulation']['initial_state']
    model = ModelIntegrator(init_state, dt, params)
    t = 0
    times=[]
    state = np.array(init_state)
    yaw = []
    ang_vel = []
    rw_i = []
    lt_torque = []
    rw_vel = []
    x = []
    y = []
    disturbance=[]
    rw_torque = []
    error = []
    momentum = []

    while t <= total_time:
        times.append(t)
        yaw.append(state[0])
        ang_vel.append(state[1])
        rw_i.append(state[2])
        lt_torque.append(state[3])
        rw_vel.append(state[4])
        x.append(state[5])
        y.append(state[6])
        disturbance.append(state[7])
        rw_torque.append(state[8])
        error.append(wrap_angle(state[0]-np.arctan2(state[6], state[5])))
        momentum.append(model.angular_momentum(state))
        state = model.rk4_step(state, t)
        t+=dt

    return pd.DataFrame({
        "time": times,
        "yaw": yaw,
        "ang_vel": ang_vel,
        "rw_i": rw_i,
        "lt_torque": lt_torque,
        "rw_vel": rw_vel,
        "x": x,
        "y": y,
        "disturbance": disturbance,
        "rw_torque": rw_torque,
        "error": error,
        "momentum": momentum,
    })


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

    df = _run_simulation(params)
    _save_results(OUTPUT_PATH, df, params)
    click.echo(f"Saved results to {OUTPUT_PATH}.")

if __name__ == "__main__":
    simulate()

