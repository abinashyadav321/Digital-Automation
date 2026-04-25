"""
main.py
~~~~~~~
Pipeline orchestrator for the Digital Twin aircraft-performance analysis.

Nepal Airlines Flight RA205  KTM → DEL  |  15 Mar 2024
Aircraft: Airbus A319-100

Run
---
    python main.py
    python main.py --data data/ra205_telemetry.csv --output output/
    python main.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add project root to sys.path so `src` is importable from any cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_telemetry,
    flight_phases,
    summary_statistics,
)
from src.flight_profile import plot_flight_profile, plot_engine_performance
from src.fuel_estimator import estimate_fuel_flow, plot_fuel_estimation, fuel_summary

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("digital_twin")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="digital_twin",
        description=(
            "Digital Twin — Aircraft Performance Analysis\n"
            "Nepal Airlines Flight RA205 (KTM → DEL)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="data/ra205_telemetry.csv",
        metavar="PATH",
        help="Path to the telemetry CSV file (default: data/ra205_telemetry.csv)",
    )
    parser.add_argument(
        "--output",
        default="output",
        metavar="DIR",
        help="Directory where output charts are saved (default: output/)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(data_path: str, output_dir: str) -> None:
    """
    Execute the full Digital Twin analytics pipeline.

    Steps
    -----
    1. Load and validate telemetry data.
    2. Compute flight phases.
    3. Print summary statistics.
    4. Generate flight-profile visualisation (4-panel).
    5. Estimate fuel flow for sensor-gap windows.
    6. Generate fuel-estimation visualisation (4-panel).
    7. Generate engine-performance chart.
    8. Print fuel-efficiency summary.
    """
    separator = "=" * 65

    # ── Step 1: Load telemetry ───────────────────────────────────────────────
    logger.info(separator)
    logger.info("STEP 1 — Loading telemetry data")
    logger.info(separator)
    df = load_telemetry(data_path)

    # ── Step 2: Flight phases ────────────────────────────────────────────────
    logger.info(separator)
    logger.info("STEP 2 — Computing flight phases")
    logger.info(separator)
    phases = flight_phases(df)
    phase_counts = phases.value_counts()
    for phase, count in phase_counts.items():
        logger.info("  %-30s  %d min", phase, count)

    # ── Step 3: Summary statistics ───────────────────────────────────────────
    logger.info(separator)
    logger.info("STEP 3 — Flight summary statistics")
    logger.info(separator)
    stats = summary_statistics(df)
    for name, val in stats["value"].items():
        logger.info("  %-40s  %s", name, val)

    # ── Step 4: Flight profile visualisation ─────────────────────────────────
    logger.info(separator)
    logger.info("STEP 4 — Generating flight profile charts")
    logger.info(separator)
    plot_flight_profile(df, phases, output_dir=output_dir)
    plot_engine_performance(df, output_dir=output_dir)

    # ── Step 5: Fuel estimation ──────────────────────────────────────────────
    logger.info(separator)
    logger.info("STEP 5 — Estimating fuel flow for sensor-gap windows")
    logger.info(separator)
    df_est = estimate_fuel_flow(df)

    # ── Step 6: Fuel estimation visualisation ────────────────────────────────
    logger.info(separator)
    logger.info("STEP 6 — Generating fuel estimation charts")
    logger.info(separator)
    plot_fuel_estimation(df_est, output_dir=output_dir)

    # ── Step 7: Engine performance chart ─────────────────────────────────────
    # (already done in step 4; no repeated call needed)

    # ── Step 8: Final fuel summary ───────────────────────────────────────────
    logger.info(separator)
    logger.info("STEP 8 — Fuel efficiency summary")
    logger.info(separator)
    fsummary = fuel_summary(df_est)
    logger.info("  Total estimated fuel burned : %.1f kg",  fsummary["total_estimated_fuel_kg"])
    logger.info("  Estimated flight distance   : %.1f nm",  fsummary["total_distance_nm"])
    logger.info("  Avg cruise efficiency       : %.4f nm/kg", fsummary["avg_cruise_efficiency_nm_per_kg"])
    logger.info("  Sensor-gap minutes filled   : %d",       fsummary["gap_minutes_filled"])

    logger.info(separator)
    logger.info("Pipeline complete.  Charts saved to: %s/", output_dir)
    logger.info(separator)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(data_path=args.data, output_dir=args.output)
