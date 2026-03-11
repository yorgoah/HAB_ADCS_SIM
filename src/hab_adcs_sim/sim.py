import numpy as np
import plotly.graph_objects as go
import click
from sim_tools.integrator import ModelIntegrator
from pathlib import Path
import json

@click.command()
@click.option(
    "--parameters",
    default= Path(r"C:\Users\abouh\OneDrive\Documents\McGill\Thesis\Simulation\config\sim_params\parameters_ec60.json"),
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
def simulate(parameters):
    with open(parameters, 'r') as f:
        params = json.load(f)
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
    lt_i = []
    rw_vel = []
    x = []
    y = []
    error = []

    while t <= total_time:
        times.append(t)
        yaw.append(state[0])
        ang_vel.append(state[1])
        rw_i.append(state[2])
        lt_i.append(state[3])
        rw_vel.append(state[4])
        x.append(state[5])
        y.append(state[6])
        error.append(state[0]-np.arctan2(state[6], state[5]))
        t+=dt
        state = model.rk4_step(state, t)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=times, y=error, name="yaw_error"))
    fig.add_trace(go.Scatter(x=times, y=ang_vel, name="ang_vel"))
    fig.add_trace(go.Scatter(x=times, y=rw_vel, name="rw_vel"))
    fig.update_layout(
        title="Simulation signals vs time",
        xaxis_title="Time [s]",
        yaxis_title="Value",
        legend_title="Signals",
    )

    fig.show()

    results = {
        "time": times,
        "yaw": yaw,
        "ang_vel": ang_vel,
        "rw_i": rw_i,
        "lt_i": lt_i,
        "rw_vel": rw_vel,
        "x": x,
        "y": y,
        "error": error
    }
    
    output_path = Path("results/simulation_results_ec60.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    simulate()

