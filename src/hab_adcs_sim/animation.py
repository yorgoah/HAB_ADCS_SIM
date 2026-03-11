import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3D

FILENAME = r"C:\Users\abouh\OneDrive\Documents\McGill\Thesis\Simulation\src\hab_adcs_sim\results\simulation_results_ec60.json"

# ---- Defaults ----
CUBE_SIZE = 10.0
FPS = 60
YAW_IN_DEGREES = False
SHOW_TRAIL = True
FORWARD_LEN_FRACTION_OF_VIEW = 0.40
FORWARD_SIGN = -1.0

SAVE_SECONDS = 10.0
OUT_MP4 = "payload_animation_10s.mp4"
OUT_GIF = "payload_animation_10s.gif"
# ------------------


def rotz(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def cube_vertices(size: float) -> np.ndarray:
    a = size / 2.0
    return np.array([
        [-a, -a, -a],
        [ a, -a, -a],
        [ a,  a, -a],
        [-a,  a, -a],
        [-a, -a,  a],
        [ a, -a,  a],
        [ a,  a,  a],
        [-a,  a,  a],
    ], dtype=float)


CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def set_axes_equal(ax):
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    xr = abs(xlim[1] - xlim[0])
    yr = abs(ylim[1] - ylim[0])
    zr = abs(zlim[1] - zlim[0])
    r = 0.5 * max(xr, yr, zr)

    xm = np.mean(xlim)
    ym = np.mean(ylim)
    zm = np.mean(zlim)
    ax.set_xlim3d(xm - r, xm + r)
    ax.set_ylim3d(ym - r, ym + r)
    ax.set_zlim3d(zm - r, zm + r)


def main():
    with open(FILENAME, "r") as f:
        data = json.load(f)

    t = np.asarray(data["time"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    z = np.asarray(data.get("z", np.zeros_like(x)), dtype=float)

    yaw = np.asarray(data["yaw"], dtype=float)
    if YAW_IN_DEGREES:
        yaw = np.deg2rad(yaw)

    n = len(t)
    if not (len(x) == len(y) == len(z) == len(yaw) == n):
        raise ValueError("time, x, y, (z), yaw must all have the same length")

    # Stride so playback is ~FPS
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0 / FPS
    sim_fps = 1.0 / dt if dt > 0 else FPS
    stride = max(1, int(round(sim_fps / FPS)))

    # Limit frames to first SAVE_SECONDS (relative to t[0])
    t_end = t[0] + SAVE_SECONDS
    last_i = int(np.searchsorted(t, t_end, side="right") - 1)
    last_i = max(0, min(last_i, n - 1))

    frame_idx = np.arange(0, last_i + 1, stride, dtype=int)
    if frame_idx.size == 0:
        frame_idx = np.array([0], dtype=int)

    V0 = cube_vertices(CUBE_SIZE)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Payload cube animation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Limits (include origin)
    pad = 0.75 * CUBE_SIZE
    xmin = min(np.min(x), 0.0) - pad
    xmax = max(np.max(x), 0.0) + pad
    ymin = min(np.min(y), 0.0) - pad
    ymax = max(np.max(y), 0.0) + pad
    zmin = min(np.min(z), 0.0) - pad
    zmax = max(np.max(z), 0.0) + pad

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    set_axes_equal(ax)

    # Long orange line based on view size
    view_span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    forward_len = FORWARD_LEN_FRACTION_OF_VIEW * view_span

    # Origin marker
    ground_station_sc = ax.scatter([0], [0], [0], s=50, marker="o", label="Ground Station/Telescope")

    # Cube edges
    edge_lines = []
    for _ in CUBE_EDGES:
        ln = Line3D([], [], [], linewidth=2, color="tab:green", label="Payload/Balloon")
        ax.add_line(ln)
        edge_lines.append(ln)

    # payload -> origin line
    pointing_ln = Line3D([], [], [], linewidth=2)
    ax.add_line(pointing_ln)

    # yaw direction line (orange)
    yaw_ln = Line3D([], [], [], linewidth=3, color="tab:orange", label="Light source Direction")
    ax.add_line(yaw_ln)

    ax.legend(handles=[ln, ground_station_sc, yaw_ln], loc="upper right")


    # Optional trail
    trail_ln = None
    if SHOW_TRAIL:
        trail_ln = Line3D([], [], [], linewidth=1)
        ax.add_line(trail_ln)
    

    # Only time text
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def update(k):
        i = frame_idx[k]
        p = np.array([x[i], y[i], z[i]], dtype=float)

        R = rotz(yaw[i])
        Vw = (V0 @ R.T) + p

        for ln, (a, b) in zip(edge_lines, CUBE_EDGES):
            ln.set_data([Vw[a, 0], Vw[b, 0]], [Vw[a, 1], Vw[b, 1]])
            ln.set_3d_properties([Vw[a, 2], Vw[b, 2]])

        # payload -> origin
        pointing_ln.set_data([p[0], 0.0], [p[1], 0.0])
        pointing_ln.set_3d_properties([p[2], 0.0])

        # yaw direction (flipped)
        fwd_body = np.array([FORWARD_SIGN * forward_len, 0.0, 0.0])
        fwd_world = fwd_body @ R.T
        p2 = p + fwd_world
        yaw_ln.set_data([p[0], p2[0]], [p[1], p2[1]])
        yaw_ln.set_3d_properties([p[2], p2[2]])

        if trail_ln is not None:
            trail_ln.set_data(x[: i + 1], y[: i + 1])
            trail_ln.set_3d_properties(z[: i + 1])

        time_text.set_text(f"t = {t[i]:.3f} s")

        if trail_ln is None:
            return (*edge_lines, pointing_ln, yaw_ln, time_text)
        return (*edge_lines, pointing_ln, yaw_ln, trail_ln, time_text)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_idx),
        interval=1000.0 / FPS,
        blit=False,
        repeat=False,
    )

    # Save 10s animation
    try:
        writer = FFMpegWriter(fps=FPS, bitrate=4000)
        anim.save(OUT_MP4, writer=writer, dpi=200)
        print(f"Saved: {OUT_MP4}")
    except Exception as e:
        print("MP4 save failed (likely missing ffmpeg). Falling back to GIF.")
        print(f"Reason: {e}")
        writer = PillowWriter(fps=FPS)
        anim.save(OUT_GIF, writer=writer)
        print(f"Saved: {OUT_GIF}")

    plt.close(fig)


if __name__ == "__main__":
    main()
