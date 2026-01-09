import numpy as np
import plotly.graph_objects as go
import click
from integrator import ModelIntegrator
from pathlib import Path
import json

@click.command()
@click.option(
    "--parameters",
    default=Path(__file__).resolve().parent / "sim_params" / "parameters.json",
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
    while t <= total_time:
        times.append(t)
        yaw.append(state[0])
        ang_vel.append(state[1])
        rw_i.append(state[2])
        lt_i.append(state[3])
        rw_vel.append(state[4])
        t+=dt
        state = model.rk4_step(state, t)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=times, y=yaw, name="yaw"))
    fig.add_trace(go.Scatter(x=times, y=ang_vel, name="ang_vel"))
    fig.add_trace(go.Scatter(x=times, y=rw_i, name="rw_i"))
    fig.add_trace(go.Scatter(x=times, y=lt_i, name="lt_i"))
    fig.add_trace(go.Scatter(x=times, y=rw_vel, name="rw_vel"))
    fig.update_layout(
        title="Simulation signals vs time",
        xaxis_title="Time [s]",
        yaxis_title="Value",
        legend_title="Signals",
    )

    fig.show()


if __name__ == "__main__":
    simulate()

