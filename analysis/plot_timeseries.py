"""Plot every logged channel of a saved simulation run as stacked time series.
"""

import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TIME_COLUMN = "time"

# Display name and unit per known channel. Unknown columns fall back to their
# own name with no unit, so a new logged channel still plots.
CHANNEL_LABELS = {
    "yaw": ("Payload yaw", "rad"),
    "ang_vel": ("Payload yaw rate", "rad/s"),
    "rw_i": ("Reaction wheel current", "A"),
    "lt_torque": ("Momentum dump torque", "N-m"),
    "rw_vel": ("Reaction wheel speed", "rad/s"),
    "x": ("Payload x position", "m"),
    "y": ("Payload y position", "m"),
    "disturbance": ("Disturbance torque", "N-m"),
    "rw_torque": ("Reaction wheel torque", "N-m"),
    "error": ("Yaw pointing error", "rad"),
    "momentum": ("Total angular momentum", "kg-m^2/s"),
}

# Single hue throughout: each panel holds one series, so color carries no
# identity and a per-panel color would only imply a distinction that is not there.
THEMES = {
    "light": {
        "series": "#2a78d6",
        "surface": "#fcfcfb",
        "primary_ink": "#0b0b0b",
        "secondary_ink": "#52514e",
        "muted_ink": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "series": "#3987e5",
        "surface": "#1a1a19",
        "primary_ink": "#ffffff",
        "secondary_ink": "#c3c2b7",
        "muted_ink": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}

ROW_HEIGHT_PX = 170
DEFAULT_MAX_POINTS = 4000


def load_results(results_path: Path) -> pd.DataFrame:
    """Read a results pickle, accepting either the cache envelope or a bare frame."""
    with results_path.open("rb") as f:
        loaded = pickle.load(f)

    if isinstance(loaded, pd.DataFrame):
        return loaded
    if isinstance(loaded, dict) and isinstance(loaded.get("data"), pd.DataFrame):
        return loaded["data"]
    raise click.ClickException(
        f"{results_path} does not hold a DataFrame or a cache envelope with a 'data' frame."
    )


def channel_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns to plot, in logged order, excluding the time base."""
    return [
        column
        for column in df.columns
        if column != TIME_COLUMN and pd.api.types.is_numeric_dtype(df[column])
    ]


def minmax_decimate(
    x: np.ndarray, y: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a series to about `max_points` samples, keeping its envelope.
    """
    n = len(x)
    if max_points <= 0 or n <= max_points:
        return x, y

    buckets = max(1, max_points // 2)
    edges = np.linspace(0, n, buckets + 1).astype(int)
    keep = [0, n - 1]
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        segment = y[start:end]
        if np.all(np.isnan(segment)):
            keep.append(start)
            continue
        keep.append(start + int(np.nanargmin(segment)))
        keep.append(start + int(np.nanargmax(segment)))

    indices = np.unique(np.asarray(keep))
    return x[indices], y[indices]


def axis_title(column: str) -> str:
    name, unit = CHANNEL_LABELS.get(column, (column, ""))
    return f"{name} [{unit}]" if unit else name


def build_figure(
    df: pd.DataFrame, theme: str, title: str, max_points: int = DEFAULT_MAX_POINTS
) -> go.Figure:
    columns = channel_columns(df)
    if not columns:
        raise click.ClickException("No numeric channels to plot.")
    if TIME_COLUMN not in df.columns:
        raise click.ClickException(f"Results have no '{TIME_COLUMN}' column to plot against.")

    colors = THEMES[theme]
    rows = len(columns)
    times = df[TIME_COLUMN].to_numpy()

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.03, 0.6 / rows),
        subplot_titles=[axis_title(column) for column in columns],
    )

    plotted_points = 0
    for row, column in enumerate(columns, start=1):
        x, y = minmax_decimate(times, df[column].to_numpy(), max_points)
        plotted_points = max(plotted_points, len(x))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=column,
                line=dict(color=colors["series"], width=2),
                hovertemplate=f"{column}: %{{y:.4g}}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    # Left-align the panel titles: centered titles read as chart titles and
    # fight the axis they sit above.
    for annotation in fig.layout.annotations:
        annotation.update(
            x=0,
            xanchor="left",
            font=dict(size=12, color=colors["secondary_ink"]),
        )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=colors["grid"],
        gridwidth=1,
        zeroline=False,
        linecolor=colors["axis"],
        tickfont=dict(color=colors["muted_ink"], size=11),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor=colors["axis"],
        spikedash="solid",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=colors["grid"],
        gridwidth=1,
        zeroline=True,
        zerolinecolor=colors["axis"],
        zerolinewidth=1,
        linecolor=colors["axis"],
        tickfont=dict(color=colors["muted_ink"], size=11),
    )
    fig.update_xaxes(title_text="Time [s]", row=rows, col=1)

    heading = title
    if plotted_points < len(times):
        heading += (
            f"<br><span style='font-size:12px;color:{colors['muted_ink']}'>"
            f"{len(times):,} samples, min/max reduced to {plotted_points:,} points per channel"
            "</span>"
        )

    fig.update_layout(
        title=dict(
            text=heading, x=0, xanchor="left", font=dict(color=colors["primary_ink"], size=18)
        ),
        height=ROW_HEIGHT_PX * rows + 140,
        showlegend=False,
        hovermode="x unified",
        paper_bgcolor=colors["surface"],
        plot_bgcolor=colors["surface"],
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=colors["secondary_ink"]),
        margin=dict(l=70, r=30, t=80, b=60),
    )
    return fig


@click.command()
@click.option(
    "--results",
    "results_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Simulation results pickle to analyse.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the figure to this HTML file instead of opening a browser window.",
)
@click.option(
    "--theme",
    type=click.Choice(sorted(THEMES)),
    default="light",
    show_default=True,
    help="Color scheme of the rendered figure.",
)
@click.option(
    "--max-points",
    default=DEFAULT_MAX_POINTS,
    show_default=True,
    help="Points drawn per channel after min/max reduction. 0 plots every sample.",
)
def plot_timeseries(
    results_path: Path, output_path: Path | None, theme: str, max_points: int
) -> None:
    """Stack every logged channel of a saved run as its own time-series panel."""
    df = load_results(results_path)
    fig = build_figure(
        df, theme, title=f"Logged channels - {results_path.stem}", max_points=max_points
    )

    if output_path is None:
        fig.show()
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    click.echo(f"Wrote {output_path}.")


if __name__ == "__main__":
    plot_timeseries()
