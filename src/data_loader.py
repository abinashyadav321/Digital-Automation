"""
data_loader.py
~~~~~~~~~~~~~~
Load, validate, and pre-process the raw telemetry CSV for Flight RA205.

Responsibilities
----------------
* Read the CSV produced by the data-capture system (or the bundled sample).
* Report data quality: total rows, columns with NaN, gap extents.
* Return a clean DataFrame ready for downstream analysis; missing numeric
  values are flagged but NOT filled here (that is the fuel-estimator's job).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "timestamp",
    "elapsed_min",
    "altitude_ft",
    "vertical_speed_fpm",
    "ias_kts",
    "tas_kts",
    "mach",
    "oat_c",
    "latitude",
    "longitude",
    "heading_deg",
    "engine1_n1_pct",
    "engine2_n1_pct",
    "fuel_flow_total_kgh",
    "fuel_remaining_kg",
]

NUMERIC_COLUMNS = [c for c in REQUIRED_COLUMNS if c != "timestamp"]

# ---------------------------------------------------------------------------
# Flight-phase thresholds
# ---------------------------------------------------------------------------
# Altitude (ft) below which the aircraft is considered on the ground / taxiing
GROUND_ALTITUDE_THRESHOLD_FT = 5_000
# Vertical speed (fpm) above which the aircraft is considered climbing/descending
CLIMB_VS_THRESHOLD_FPM = 300
# Elapsed minutes at which the aircraft becomes airborne (for statistics)
AIRBORNE_START_MIN = 2
# Cruise phase window (elapsed minutes) used for fuel-flow and efficiency stats
CRUISE_START_MIN = 25
CRUISE_END_MIN = 65


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_telemetry(csv_path: str | Path) -> pd.DataFrame:
    """
    Load telemetry from *csv_path* and return a validated DataFrame.

    Parameters
    ----------
    csv_path : str or Path
        Path to the telemetry CSV file.

    Returns
    -------
    pd.DataFrame
        Index is ``elapsed_min`` (integer minutes from departure).
        ``timestamp`` is parsed to ``datetime64``.
        All numeric columns remain as ``float64``; intentional gaps are kept
        as ``NaN`` so the fuel-estimator can fill them.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If required columns are missing from the file.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {csv_path}")

    logger.info("Loading telemetry from %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # ---- validate columns --------------------------------------------------
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Telemetry file is missing columns: {missing}")

    # ---- cast numerics -------------------------------------------------------
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("elapsed_min")

    # ---- report quality ------------------------------------------------------
    _report_quality(df)

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report_quality(df: pd.DataFrame) -> None:
    """Log a data-quality summary to the console."""
    total = len(df)
    logger.info("Telemetry rows: %d  |  time span: 0 – %d min", total, df.index.max())

    nan_summary = df.isna().sum()
    nan_cols = nan_summary[nan_summary > 0]

    if nan_cols.empty:
        logger.info("No NaN values detected – all sensors nominal.")
    else:
        logger.warning("Sensor gaps detected:")
        for col, count in nan_cols.items():
            gap_indices = df.index[df[col].isna()].tolist()
            logger.warning(
                "  %-30s  %3d NaN  (minutes: %s … %s)",
                col,
                count,
                gap_indices[0],
                gap_indices[-1],
            )


def flight_phases(df: pd.DataFrame) -> pd.Series:
    """
    Assign a named flight phase label to every row based on altitude and
    vertical speed.

    Returns
    -------
    pd.Series
        String phase label indexed the same as *df*.
    """
    alt = df["altitude_ft"]
    vs = df["vertical_speed_fpm"]

    conditions = [
        (alt < GROUND_ALTITUDE_THRESHOLD_FT) & (vs.abs() < CLIMB_VS_THRESHOLD_FPM),  # ground / taxi
        (alt < 10_000) & (vs > CLIMB_VS_THRESHOLD_FPM),                               # takeoff / initial climb
        vs > CLIMB_VS_THRESHOLD_FPM,                                                   # climb
        vs < -CLIMB_VS_THRESHOLD_FPM,                                                  # descent
        (alt > 10_000) & (vs.abs() <= CLIMB_VS_THRESHOLD_FPM),                        # cruise
    ]
    labels = ["Ground", "Takeoff/Initial Climb", "Climb", "Descent", "Cruise"]
    phase = pd.Series("Unknown", index=df.index)
    for cond, label in zip(conditions, labels):
        phase[cond] = label
    return phase


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of key performance statistics for the flight.

    Returns
    -------
    pd.DataFrame
        Rows are stat names, single column ``value``.
    """
    airborne = df[df.index >= AIRBORNE_START_MIN]

    stats = {
        "Max Altitude (ft)":              df["altitude_ft"].max(),
        "Max IAS (kts)":                  df["ias_kts"].max(),
        "Max TAS (kts)":                  df["tas_kts"].max(),
        "Max Mach":                       df["mach"].max(),
        "Min OAT (°C)":                   df["oat_c"].min(),
        "Total flight time (min)":        int(df.index.max()),
        "Fuel burned (kg)":               round(
            df["fuel_remaining_kg"].iloc[0] - df["fuel_remaining_kg"].iloc[-1], 1
        ),
        "Fuel flow NaN count":            int(df["fuel_flow_total_kgh"].isna().sum()),
        "Avg cruise fuel flow (kg/hr)":   round(
            airborne.loc[CRUISE_START_MIN:CRUISE_END_MIN, "fuel_flow_total_kgh"].mean(), 1
        ),
    }
    return pd.DataFrame.from_dict(stats, orient="index", columns=["value"])
