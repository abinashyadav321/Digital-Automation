"""
flight_profile.py
~~~~~~~~~~~~~~~~~
Visualise the flight profile of Nepal Airlines Flight RA205.

Produces a single figure with four subplots:
  1. Altitude vs time (with flight-phase colour bands)
  2. Airspeed (IAS & TAS) vs time
  3. Vertical speed vs time
  4. Flight path (lat / lon ground track)

Usage
-----
    from src.flight_profile import plot_flight_profile
    fig = plot_flight_profile(df, output_dir="output")
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")        # non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------
# Maximum altitude shown on the y-axis of the altitude profile (thousands of ft).
# Set slightly above FL350 (35 000 ft = 35k) to give visual headroom.
MAX_ALTITUDE_DISPLAY_KFT = 40

# ---------------------------------------------------------------------------
# Colour scheme for flight phases
# ---------------------------------------------------------------------------
PHASE_COLORS = {
    "Ground":                "#a8d8a8",   # soft green
    "Takeoff/Initial Climb": "#ffd580",   # amber
    "Climb":                 "#80bfff",   # sky blue
    "Cruise":                "#c8a0d8",   # lavender
    "Descent":               "#ffb380",   # peach
    "Unknown":               "#eeeeee",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_flight_profile(
    df: pd.DataFrame,
    phases: pd.Series,
    output_dir: str | Path = "output",
    filename: str = "flight_profile.png",
) -> plt.Figure:
    """
    Generate a four-panel flight-profile figure and save it to *output_dir*.

    Parameters
    ----------
    df : pd.DataFrame
        Telemetry DataFrame indexed by ``elapsed_min``.
    phases : pd.Series
        Phase labels (same index as *df*), from
        :func:`src.data_loader.flight_phases`.
    output_dir : str or Path
        Directory where the PNG will be saved (created if needed).
    filename : str
        Output file name (PNG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(15, 10),
        facecolor="#f8f9fa",
        gridspec_kw={"hspace": 0.38, "wspace": 0.28},
    )
    fig.suptitle(
        "Digital Twin — Nepal Airlines Flight RA205  (KTM → DEL)\n"
        "Aircraft: Airbus A319-100  |  15 Mar 2024",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color="#1a1a2e",
    )

    t = df.index.to_numpy(dtype=float)

    # ---- helper: shade phase bands -----------------------------------------
    def shade_phases(ax):
        prev_phase = None
        start = t[0]
        for i, (ti, ph) in enumerate(zip(t, phases)):
            if ph != prev_phase:
                if prev_phase is not None:
                    ax.axvspan(start, ti, alpha=0.18,
                               color=PHASE_COLORS.get(prev_phase, "#eeeeee"),
                               zorder=0)
                start = ti
                prev_phase = ph
        if prev_phase is not None:
            ax.axvspan(start, t[-1], alpha=0.18,
                       color=PHASE_COLORS.get(prev_phase, "#eeeeee"),
                       zorder=0)

    # ---- 1. Altitude -------------------------------------------------------
    ax1 = axes[0, 0]
    shade_phases(ax1)
    ax1.plot(t, df["altitude_ft"] / 1000, color="#1565c0", linewidth=2, zorder=3)
    ax1.set_xlabel("Elapsed time (min)", fontsize=10)
    ax1.set_ylabel("Altitude (×1000 ft)", fontsize=10)
    ax1.set_title("Altitude Profile", fontweight="bold")
    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylim(0, MAX_ALTITUDE_DISPLAY_KFT)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}k"))
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.tick_params(axis="both", labelsize=9)

    # legend patches
    legend_patches = [
        mpatches.Patch(color=c, alpha=0.5, label=p)
        for p, c in PHASE_COLORS.items() if p != "Unknown"
    ]
    ax1.legend(handles=legend_patches, fontsize=7.5, loc="upper right",
               framealpha=0.85, ncol=2)

    # ---- 2. Airspeed -------------------------------------------------------
    ax2 = axes[0, 1]
    shade_phases(ax2)
    ax2.plot(t, df["ias_kts"], color="#c62828", linewidth=2, label="IAS (kts)", zorder=3)
    ax2.plot(t, df["tas_kts"], color="#0277bd", linewidth=2,
             linestyle="--", label="TAS (kts)", zorder=3)

    ax2_r = ax2.twinx()
    ax2_r.plot(t, df["mach"], color="#558b2f", linewidth=1.5,
               linestyle=":", label="Mach", zorder=2)
    ax2_r.set_ylabel("Mach", fontsize=10, color="#558b2f")
    ax2_r.tick_params(axis="y", labelcolor="#558b2f", labelsize=9)
    ax2_r.set_ylim(0, 1.0)

    ax2.set_xlabel("Elapsed time (min)", fontsize=10)
    ax2.set_ylabel("Speed (kts)", fontsize=10)
    ax2.set_title("Airspeed Profile", fontweight="bold")
    ax2.set_xlim(t[0], t[-1])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.tick_params(axis="both", labelsize=9)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left",
               framealpha=0.85)

    # ---- 3. Vertical speed -------------------------------------------------
    ax3 = axes[1, 0]
    shade_phases(ax3)
    vs = df["vertical_speed_fpm"].to_numpy(dtype=float)
    ax3.fill_between(t, vs, where=vs >= 0, color="#1976d2", alpha=0.55,
                     label="Climb", zorder=3)
    ax3.fill_between(t, vs, where=vs < 0, color="#e53935", alpha=0.55,
                     label="Descent", zorder=3)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xlabel("Elapsed time (min)", fontsize=10)
    ax3.set_ylabel("Vertical Speed (fpm)", fontsize=10)
    ax3.set_title("Vertical Speed", fontweight="bold")
    ax3.set_xlim(t[0], t[-1])
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax3.tick_params(axis="both", labelsize=9)

    # ---- 4. Flight path (ground track) ------------------------------------
    ax4 = axes[1, 1]
    lats = df["latitude"].to_numpy(dtype=float)
    lons = df["longitude"].to_numpy(dtype=float)

    # colour track by altitude
    points = np.array([lons, lats]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    from matplotlib.collections import LineCollection
    from matplotlib.cm import get_cmap
    norm = plt.Normalize(vmin=df["altitude_ft"].min(),
                         vmax=df["altitude_ft"].max())
    lc = LineCollection(segments, cmap="plasma", norm=norm, linewidth=2.5)
    lc.set_array(df["altitude_ft"].to_numpy(dtype=float)[:-1])
    ax4.add_collection(lc)
    plt.colorbar(lc, ax=ax4, label="Altitude (ft)", pad=0.02)

    # origin / destination markers
    ax4.plot(lon_ktm := 85.3591, lat_ktm := 27.6966,
             "go", markersize=10, label="KTM", zorder=5)
    ax4.plot(lon_del := 77.1000, lat_del := 28.5562,
             "rs", markersize=10, label="DEL", zorder=5)
    ax4.annotate("KTM", xy=(lon_ktm, lat_ktm), xytext=(0.5, 0.5),
                 textcoords="offset points", fontsize=9, fontweight="bold")
    ax4.annotate("DEL", xy=(lon_del, lat_del), xytext=(0.5, 0.5),
                 textcoords="offset points", fontsize=9, fontweight="bold")

    ax4.set_xlim(lons.min() - 0.3, lons.max() + 0.3)
    ax4.set_ylim(lats.min() - 0.2, lats.max() + 0.2)
    ax4.set_xlabel("Longitude (°E)", fontsize=10)
    ax4.set_ylabel("Latitude (°N)", fontsize=10)
    ax4.set_title("Ground Track (colour = altitude)", fontweight="bold")
    ax4.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.tick_params(axis="both", labelsize=9)

    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Flight profile saved → %s", out_path)
    return fig


def plot_engine_performance(
    df: pd.DataFrame,
    output_dir: str | Path = "output",
    filename: str = "engine_performance.png",
) -> plt.Figure:
    """
    Plot engine N1 fan speeds over the flight.

    Parameters
    ----------
    df : pd.DataFrame
        Telemetry DataFrame indexed by ``elapsed_min``.
    output_dir : str or Path
        Directory where the PNG will be saved.
    filename : str
        Output file name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#f8f9fa")
    fig.suptitle(
        "Engine Performance — Nepal Airlines RA205",
        fontsize=13, fontweight="bold", color="#1a1a2e",
    )

    t = df.index.to_numpy(dtype=float)
    ax.plot(t, df["engine1_n1_pct"], color="#1565c0", linewidth=1.8,
            label="Engine 1 N1 (%)", zorder=3)
    ax.plot(t, df["engine2_n1_pct"], color="#c62828", linewidth=1.8,
            linestyle="--", label="Engine 2 N1 (%)", zorder=3)

    ax.set_xlabel("Elapsed time (min)", fontsize=10)
    ax.set_ylabel("N1 Fan Speed (%)", fontsize=10)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", labelsize=9)

    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Engine performance plot saved → %s", out_path)
    return fig
