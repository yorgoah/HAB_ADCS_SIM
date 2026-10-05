"""Figures for the README.

    python -m hab_adcs_sim.figures ascent WHEEL_RUN.pkl PIVOT_RUN.pkl
    python -m hab_adcs_sim.figures monte-carlo monte_carlo/nominal_wheel_only monte_carlo/nominal_pivot_motor
    python -m hab_adcs_sim.figures animate WHEEL_RUN.pkl --start 1074 --duration 32
"""
import pickle
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Wedge
from matplotlib.transforms import Affine2D

from hab_adcs_sim.monte_carlo.per_run_analysis import (
    ERROR_BOUND_DEG,
    RW_SATURATION_MARGIN_RPM,
    RW_SATURATION_RPM,
    pct_time_within_bound,
    pointing_windows,
)
from sim_tools.pointing import PointingState

FIGURES_DIR = Path(__file__).resolve().parents[2] / "docs" / "figures"

WHEEL_COLOR = "#1f5fbf"
PIVOT_COLOR = "#d9541e"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#cde2fb"  # pointing bound
SATURATED_SHADE = "#fde3a7"
STABILIZE_SHADE = "#ecebe7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load_run(path):
    with open(path, "rb") as f:
        result = pickle.load(f)
    return result["parameters"], result["data"]


def minmax_decimate(x, y, bins):
    # Min and max of every bin, so short spikes still show
    n = len(x) // bins * bins
    x = x[:n].reshape(bins, -1)
    y = y[:n].reshape(bins, -1)
    xs = np.repeat(x.mean(axis=1), 2)
    ys = np.column_stack((y.min(axis=1), y.max(axis=1))).ravel()
    return xs, ys


def break_wraps(y, jump=180.0):
    # NaN where the wrapped error jumps from +180 to -180, so no line is drawn across
    y = y.astype(float).copy()
    y[1:][np.abs(np.diff(y)) > jump] = np.nan
    return y


def error_bound_band(ax):
    ax.axhspan(-ERROR_BOUND_DEG, ERROR_BOUND_DEG, color=BAND, lw=0, zorder=0.5)


def ascent_figure(runs, output):
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 2.6 * len(runs) + 0.6), sharex=True, layout="constrained")
    for ax, (label, color, params, df) in zip(axes, runs):
        t_min = df["time"].to_numpy() / 60
        error = np.rad2deg(df["error"].to_numpy())
        within = pct_time_within_bound(df, params)
        windows = int(pointing_windows(df, params))

        error_bound_band(ax)
        ax.plot(*minmax_decimate(t_min, error, 3000), color=color, lw=0.7)
        ax.set_ylim(-180, 180)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_ylabel("Azimuth error (°)")
        ax.set_title(label, loc="left", color=INK, fontweight="semibold")
        ax.set_title(f"{within:.0f}% of the ascent within ±{ERROR_BOUND_DEG:g}°, {windows} windows of 30 s",
                     loc="right", color=INK_2)

    wind = runs[0][2]["wind_params"]
    start_km, rate_km = wind["start_altitude_m"] / 1000, wind["ascent_rate_m_s"] * 60 / 1000
    top = axes[0].secondary_xaxis("top", functions=(lambda t: start_km + rate_km * t,
                                                    lambda h: (h - start_km) / rate_km))
    top.set_xlabel("Altitude (km)")
    top.set_xticks([5, 10, 15, 20])
    top.tick_params(colors=MUTED, labelcolor=INK_2)
    axes[-1].set_xlabel("Time since launch (min)")
    axes[-1].set_xlim(0, t_min[-1])
    fig.savefig(output, dpi=200)
    plt.close(fig)


def load_campaign_metrics(campaign_dir):
    runs = []
    for path in sorted(Path(campaign_dir).glob("*/run_*/analysis.pkl")):
        with path.open("rb") as f:
            runs.append(pickle.load(f))
    if not runs:
        raise click.ClickException(f"No analysed runs in {campaign_dir}")
    metrics = {name: np.array([r["metrics"][name] for r in runs]) for name in runs[0]["metrics"]}
    return metrics, runs[0]["limits"]


def monte_carlo_figure(campaigns, output, run_s=600.0):
    limits = campaigns[0][2]
    bound, window_s = limits["error_bound_deg"], limits["pointing_window_s"]
    n_windows = int(run_s // window_s)
    saturated_max = max(m["rw_saturated_pct"].max() for _, _, _, m in campaigns)
    panels = [
        ("pct_time_within_bound", np.arange(0, 105, 5), f"Time within ±{bound:g}° (% of run)"),
        ("rw_saturated_pct", np.arange(0, 2.5 * np.ceil(saturated_max / 2.5 + 1e-9) + 2.5, 2.5),
         "Time with the wheel saturated (% of run)"),
        ("pointing_windows", np.arange(-0.5, n_windows + 1.5, 1), f"{window_s:g} s pointing windows per run"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True, layout="constrained")
    for ax, (metric, bins, xlabel) in zip(axes, panels):
        for label, color, _, metrics in campaigns:
            values = metrics[metric]
            ax.hist(values, bins=bins, histtype="stepfilled", facecolor=color, alpha=0.22, lw=0)
            ax.hist(values, bins=bins, histtype="step", edgecolor=color, lw=1.5, label=label)
        ax.set_xlabel(xlabel)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(axis="x", visible=False)
    axes[2].set_xticks(np.arange(0, n_windows + 1, 4))
    axes[0].set_ylabel(f"Runs (of {len(campaigns[0][3]['pointing_windows'])})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(campaigns))
    fig.savefig(output, dpi=200)
    plt.close(fig)


PHASES = {
    PointingState.POINTING.value: "Pointing at the ground station",
    PointingState.SATURATED.value: "Wheel saturated: spinning it back down",
    PointingState.STABILIZE.value: "Stabilizing the spin",
}


def shade_phases(ax, t, state, label=False):
    edges = np.flatnonzero(np.diff(state)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(state)]))
    for a, b in zip(starts, ends):
        shade = {PointingState.SATURATED.value: SATURATED_SHADE,
                 PointingState.STABILIZE.value: STABILIZE_SHADE}.get(int(state[a]))
        if shade is None:
            continue
        ax.axvspan(t[a], t[b - 1], color=shade, lw=0, zorder=0)
        if label:
            name = "desaturate" if state[a] == PointingState.SATURATED.value else "stabilize"
            ax.text((t[a] + t[b - 1]) / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=INK_2)


def pointing_animation(params, df, start, duration, output, speed=2.0, fps=15):
    window = df[(df["time"] >= start) & (df["time"] <= start + duration)].iloc[::10]
    t = window["time"].to_numpy() - start
    error = np.rad2deg(window["error"].to_numpy())
    rpm = window["rw_vel"].to_numpy() * 30 / np.pi
    state = window["pointing_state"].to_numpy().astype(int)
    altitude_km = window["altitude"].to_numpy() / 1000
    frames = np.arange(0.0, t[-1], speed / fps)
    frames = np.concatenate((frames, np.full(fps, frames[-1])))  # pause before looping

    fig = plt.figure(figsize=(10, 4.2))
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.7), left=0.02, right=0.98, bottom=0.13, top=0.9,
                            wspace=0.12, hspace=0.35)
    view = fig.add_subplot(grid[:, 0])
    err_ax = fig.add_subplot(grid[0, 1])
    rpm_ax = fig.add_subplot(grid[1, 1], sharex=err_ax)

    # Top-down view with the ground station fixed to the right
    view.set_xlim(-1.05, 1.45)
    view.set_ylim(-1.25, 1.25)
    view.set_aspect("equal")
    view.axis("off")
    view.add_patch(Wedge((0, 0), 1.25, -ERROR_BOUND_DEG, ERROR_BOUND_DEG, color=BAND, lw=0))
    view.plot([0.5, 1.28], [0, 0], color=MUTED, lw=0.8, ls=(0, (4, 3)))
    view.plot(1.33, 0, marker="v", color=INK_2, ms=9)
    view.text(1.33, -0.12, "ground\nstation", ha="center", va="top", fontsize=8, color=INK_2)
    gondola = Rectangle((-0.4, -0.4), 0.8, 0.8, facecolor="#ffffff", edgecolor=INK, lw=1.5)
    view.add_patch(gondola)
    beam = Wedge((0, 0), 1.15, -2.5, 2.5, color=WHEEL_COLOR, lw=0)
    view.add_patch(beam)
    wheel = view.add_patch(plt.Circle((0, 0), 0.16, facecolor="#ffffff", edgecolor=INK_2, lw=1.2))
    phase_text = fig.text(0.19, 0.95, "", ha="center", va="top", fontsize=10, color=INK, fontweight="semibold")
    clock_text = fig.text(0.19, 0.04, "", ha="center", va="bottom", fontsize=9, color=INK_2)

    shade_phases(err_ax, t, state, label=True)
    error_bound_band(err_ax)
    err_ax.set_ylim(-180, 180)
    err_ax.set_yticks([-180, -90, 0, 90, 180])
    err_ax.set_ylabel("Azimuth error (°)")
    err_ax.tick_params(labelbottom=False)
    err_ax.text(t[-1], ERROR_BOUND_DEG + 4, f"±{ERROR_BOUND_DEG:g}°", ha="right", va="bottom", fontsize=8,
                color=INK_2)
    err_line, = err_ax.plot([], [], color=WHEEL_COLOR, lw=1.5)
    err_dot, = err_ax.plot([], [], "o", color=WHEEL_COLOR, ms=5)

    shade_phases(rpm_ax, t, state)
    limit = RW_SATURATION_RPM - RW_SATURATION_MARGIN_RPM
    rpm_ax.axhline(limit, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    rpm_ax.text(0.3, limit - 60, "saturation", ha="left", va="top", fontsize=8, color=INK_2)
    rpm_ax.set_ylim(0, 3200)
    rpm_ax.set_yticks([0, 1500, 3000])
    rpm_ax.set_ylabel("Wheel speed (rpm)")
    rpm_ax.set_xlabel("Time (s)")
    rpm_ax.set_xlim(0, t[-1])
    rpm_line, = rpm_ax.plot([], [], color=WHEEL_COLOR, lw=1.5)
    rpm_dot, = rpm_ax.plot([], [], "o", color=WHEEL_COLOR, ms=5)
    error_trace = break_wraps(error)

    def draw(now):
        i = min(np.searchsorted(t, now, side="right"), len(t)) - 1
        angle = error[i]
        on_target = abs(angle) <= ERROR_BOUND_DEG
        rotation = Affine2D().rotate_deg(angle) + view.transData
        gondola.set_transform(rotation)
        beam.set_theta1(angle - 2.5)
        beam.set_theta2(angle + 2.5)
        beam.set_color(WHEEL_COLOR if on_target else MUTED)
        wheel.set_facecolor(SATURATED_SHADE if state[i] == PointingState.SATURATED.value else "#ffffff")
        phase_text.set_text(PHASES[state[i]])
        clock_text.set_text(f"t = {t[i]:4.1f} s    altitude {altitude_km[i]:.1f} km    {speed:g}x speed")
        err_line.set_data(t[: i + 1], error_trace[: i + 1])
        err_dot.set_data([t[i]], [angle])
        rpm_line.set_data(t[: i + 1], rpm[: i + 1])
        rpm_dot.set_data([t[i]], [rpm[i]])
        return gondola, beam, wheel, phase_text, clock_text, err_line, err_dot, rpm_line, rpm_dot

    animation = FuncAnimation(fig, draw, frames=frames, blit=False)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


@click.group()
def cli():
    """Make the README figures."""


@cli.command()
@click.argument("wheel_run", type=click.Path(exists=True, dir_okay=False))
@click.argument("pivot_run", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", type=click.Path(dir_okay=False), default=FIGURES_DIR / "ascent_wheel_vs_pivot.png")
def ascent(wheel_run, pivot_run, output):
    """Azimuth error over an ascent, wheel only against wheel and pivot motor."""
    runs = [("Reaction wheel only", WHEEL_COLOR, *load_run(wheel_run)),
            ("Reaction wheel and pivot motor", PIVOT_COLOR, *load_run(pivot_run))]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    ascent_figure(runs, output)
    click.echo(f"Saved {output}")


@cli.command("monte-carlo")
@click.argument("wheel_campaign", type=click.Path(exists=True, file_okay=False))
@click.argument("pivot_campaign", type=click.Path(exists=True, file_okay=False))
@click.option("--output", type=click.Path(dir_okay=False), default=FIGURES_DIR / "monte_carlo.png")
def monte_carlo(wheel_campaign, pivot_campaign, output):
    """Histograms of the per-run metrics of two campaigns."""
    campaigns = []
    for label, color, path in (("Reaction wheel only", WHEEL_COLOR, wheel_campaign),
                               ("Reaction wheel and pivot motor", PIVOT_COLOR, pivot_campaign)):
        metrics, limits = load_campaign_metrics(path)
        campaigns.append((label, color, limits, metrics))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    monte_carlo_figure(campaigns, output)
    click.echo(f"Saved {output}")


@cli.command()
@click.argument("run", type=click.Path(exists=True, dir_okay=False))
@click.option("--start", type=float, required=True, help="Start of the window (s).")
@click.option("--duration", type=float, default=30.0, show_default=True)
@click.option("--speed", type=float, default=2.0, show_default=True, help="Playback speed.")
@click.option("--output", type=click.Path(dir_okay=False), default=FIGURES_DIR / "pointing_recovery.gif")
def animate(run, start, duration, speed, output):
    """Animated GIF of a window of a run."""
    params, df = load_run(run)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pointing_animation(params, df, start, duration, output, speed=speed)
    click.echo(f"Saved {output}")


if __name__ == "__main__":
    cli()
