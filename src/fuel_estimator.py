"""
fuel_estimator.py
~~~~~~~~~~~~~~~~~
Estimate fuel flow / fuel efficiency for time windows where the physical
sensor data is unavailable (NaN gaps in ``fuel_flow_total_kgh``).

Two complementary approaches are implemented:

1. **Physics-based (Breguet-inspired) model**
   Fuel flow is modelled as a function of engine N1, true airspeed, altitude,
   and OAT.  Coefficients are fitted on the known (non-NaN) rows using
   ordinary least-squares regression so the model is always grounded in the
   measured flight data.

2. **Machine-learning model (Gradient Boosting)**
   A ``GradientBoostingRegressor`` is trained on the same known rows.  This
   captures non-linearities (climb / cruise / descent) more accurately.

The pipeline uses both models: the physics model provides an interpretable
baseline, and the ML model is used to fill the actual NaN gaps.  A
comparison chart is written to *output/*.

Public functions
----------------
estimate_fuel_flow(df)  → pd.DataFrame
    Returns the input DataFrame with two new columns:
    ``fuel_flow_physics_kgh``  and  ``fuel_flow_estimated_kgh``.

plot_fuel_estimation(df, …)  → fig
    Four-panel chart: measured vs estimated fuel flow, residuals, feature
    importance, and fuel efficiency.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler

from src.data_loader import CRUISE_START_MIN, CRUISE_END_MIN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants — fuel-flow and efficiency thresholds
# ---------------------------------------------------------------------------
MIN_FUEL_FLOW_KGH = 300       # physical lower bound for clipping predictions
MAX_FUEL_FLOW_KGH = 6_500     # physical upper bound (≈ TOGA on A319)
MIN_TRAINING_SAMPLES = 10     # minimum known rows needed to train the estimator

# Efficiency-plot masks: exclude near-ground and idle-descent segments
AIRBORNE_IAS_THRESHOLD_KTS = 120    # below this IAS the efficiency metric is uninformative
IDLE_DESCENT_VS_THRESHOLD_FPM = -400  # below this VS (descent) efficiency is not meaningful

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURES = [
    "altitude_ft",
    "tas_kts",
    "mach",
    "vertical_speed_fpm",
    "oat_c",
    "engine1_n1_pct",
    "engine2_n1_pct",
]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature matrix derived from telemetry columns."""
    X = df[FEATURES].copy()

    # interaction / non-linear terms that help the physics model
    X["n1_mean"] = (df["engine1_n1_pct"] + df["engine2_n1_pct"]) / 2.0
    X["n1_sq"] = X["n1_mean"] ** 2
    X["alt_kft"] = df["altitude_ft"] / 1000.0
    X["tas_sq"] = df["tas_kts"] ** 2

    return X


# ---------------------------------------------------------------------------
# Physics-based model
# ---------------------------------------------------------------------------

def _fit_physics_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[LinearRegression, StandardScaler]:
    """
    Fit a linear regression on physics-inspired features.

    Returns
    -------
    (model, scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_scaled, y_train)
    return model, scaler


# ---------------------------------------------------------------------------
# ML model
# ---------------------------------------------------------------------------

def _fit_ml_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GradientBoostingRegressor:
    """
    Fit a Gradient Boosting regressor on the training rows.

    Returns
    -------
    GradientBoostingRegressor
    """
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_fuel_flow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate fuel flow for every row in *df*, returning an augmented copy.

    Two new columns are added:

    * ``fuel_flow_physics_kgh``  — linear physics-based estimate (all rows)
    * ``fuel_flow_estimated_kgh`` — Gradient Boosting estimate used to fill
      NaN gaps; measured values are preserved for non-gap rows.

    Parameters
    ----------
    df : pd.DataFrame
        Telemetry DataFrame indexed by ``elapsed_min`` (from data_loader).

    Returns
    -------
    pd.DataFrame
        Augmented copy of *df*.
    """
    df = df.copy()

    X_all = _build_features(df)

    # ---- split into train (known) vs gap (NaN) rows ----------------------
    known_mask = df["fuel_flow_total_kgh"].notna()
    gap_mask = ~known_mask

    n_known = known_mask.sum()
    n_gap = gap_mask.sum()
    logger.info(
        "Fuel-flow: %d measured rows, %d NaN rows to estimate", n_known, n_gap
    )

    if n_known < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Only {n_known} measured fuel-flow rows — cannot train estimator."
        )

    X_train = X_all[known_mask]
    y_train = df.loc[known_mask, "fuel_flow_total_kgh"]

    # ---- physics model -----------------------------------------------------
    phys_model, scaler = _fit_physics_model(X_train, y_train)
    X_all_scaled = scaler.transform(X_all)
    df["fuel_flow_physics_kgh"] = np.clip(
        phys_model.predict(X_all_scaled), MIN_FUEL_FLOW_KGH, MAX_FUEL_FLOW_KGH
    )

    phys_train_pred = phys_model.predict(scaler.transform(X_train))
    logger.info(
        "Physics model  — MAE: %.1f kg/hr  R²: %.4f",
        mean_absolute_error(y_train, phys_train_pred),
        r2_score(y_train, phys_train_pred),
    )

    # ---- ML model ----------------------------------------------------------
    ml_model = _fit_ml_model(X_train, y_train)
    ml_full_pred = np.clip(ml_model.predict(X_all), MIN_FUEL_FLOW_KGH, MAX_FUEL_FLOW_KGH)

    # cross-validated predictions for honest out-of-fold MAE / residuals
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    ml_cv_pred = np.clip(
        cross_val_predict(
            GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.08,
                subsample=0.85, random_state=42,
            ),
            X_train, y_train, cv=cv,
        ),
        MIN_FUEL_FLOW_KGH, MAX_FUEL_FLOW_KGH,
    )
    cv_mae = mean_absolute_error(y_train, ml_cv_pred)
    cv_r2 = r2_score(y_train, ml_cv_pred)
    logger.info(
        "ML model (GBR) — CV-MAE: %.1f kg/hr  CV-R²: %.4f",
        cv_mae, cv_r2,
    )

    # store cv residuals for the plot (indexed to known rows in original df)
    cv_residuals = pd.Series(
        ml_cv_pred - y_train.to_numpy(), index=y_train.index, name="cv_residual"
    )

    # ---- assemble estimated column ----------------------------------------
    # Use measured values where available; fill gaps with ML predictions
    df["fuel_flow_estimated_kgh"] = df["fuel_flow_total_kgh"].copy()
    df.loc[gap_mask, "fuel_flow_estimated_kgh"] = ml_full_pred[gap_mask]

    # ---- compute fuel efficiency (nm / kg) --------------------------------
    # Instantaneous: TAS [kts] = nm/hr; fuel flow [kg/hr]
    # → efficiency = TAS / fuel_flow  [nm/kg]
    df["fuel_efficiency_nm_per_kg"] = np.where(
        df["fuel_flow_estimated_kgh"] > 0,
        df["tas_kts"] / df["fuel_flow_estimated_kgh"],
        np.nan,
    )

    # ---- store feature importances for plotting --------------------------
    df.attrs["feature_importances"] = dict(
        zip(X_all.columns, ml_model.feature_importances_)
    )
    df.attrs["gap_minutes"] = df.index[gap_mask].tolist()
    df.attrs["n_gap"] = int(n_gap)
    df.attrs["cv_residuals"] = cv_residuals
    df.attrs["cv_mae"] = cv_mae

    return df


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_fuel_estimation(
    df: pd.DataFrame,
    output_dir: str | Path = "output",
    filename: str = "fuel_estimation.png",
) -> plt.Figure:
    """
    Four-panel chart comparing measured vs estimated fuel flow.

    Panels
    ------
    1. Fuel flow: measured (with gaps) vs ML estimate vs physics estimate
    2. Residuals on the training rows
    3. Feature importance (Gradient Boosting)
    4. Fuel efficiency (nm / kg) over time

    Parameters
    ----------
    df : pd.DataFrame
        Augmented DataFrame returned by :func:`estimate_fuel_flow`.
    output_dir : str or Path
        Output directory.
    filename : str
        PNG file name.

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
        gridspec_kw={"hspace": 0.40, "wspace": 0.30},
    )
    fig.suptitle(
        "Digital Twin — Fuel Efficiency Estimation  (RA205 KTM → DEL)\n"
        "Filling sensor gaps using Physics-based & Gradient Boosting models",
        fontsize=13, fontweight="bold", y=0.98, color="#1a1a2e",
    )

    t = df.index.to_numpy(dtype=float)
    gap_min = df.attrs.get("gap_minutes", [])

    # ---- 1. Fuel flow comparison ------------------------------------------
    ax1 = axes[0, 0]

    # shade the NaN gap windows
    if gap_min:
        _shade_gaps(ax1, gap_min, t[-1])

    ax1.plot(t, df["fuel_flow_total_kgh"],
             color="#1565c0", linewidth=2, label="Measured", zorder=4)
    ax1.plot(t, df["fuel_flow_estimated_kgh"],
             color="#e65100", linewidth=2, linestyle="--",
             label="ML Estimate (GBR)", zorder=3)
    ax1.plot(t, df["fuel_flow_physics_kgh"],
             color="#558b2f", linewidth=1.5, linestyle=":",
             label="Physics Estimate", zorder=2)

    ax1.set_xlabel("Elapsed time (min)", fontsize=10)
    ax1.set_ylabel("Fuel Flow (kg/hr)", fontsize=10)
    ax1.set_title("Fuel Flow: Measured vs Estimated", fontweight="bold")
    ax1.set_xlim(t[0], t[-1])
    ax1.legend(fontsize=9, framealpha=0.85)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.tick_params(axis="both", labelsize=9)

    # ---- 2. Cross-validated residuals (honest out-of-fold error) ----------
    ax2 = axes[0, 1]
    cv_residuals = df.attrs.get("cv_residuals", pd.Series(dtype=float))
    cv_mae = df.attrs.get("cv_mae", float("nan"))
    if not cv_residuals.empty:
        ax2.bar(
            cv_residuals.index, cv_residuals.values,
            color=np.where(cv_residuals.values >= 0, "#1976d2", "#e53935"),
            alpha=0.75, width=0.7,
        )
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Elapsed time (min)", fontsize=10)
    ax2.set_ylabel("Residual (kg/hr)", fontsize=10)
    ax2.set_title("Cross-Validated Residuals (GBR, 5-fold)", fontweight="bold")
    ax2.set_xlim(t[0], t[-1])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.tick_params(axis="both", labelsize=9)

    # annotate cross-validated MAE
    ax2.annotate(
        f"CV-MAE = {cv_mae:.1f} kg/hr",
        xy=(0.97, 0.93), xycoords="axes fraction",
        ha="right", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
    )

    # ---- 3. Feature importance -------------------------------------------
    ax3 = axes[1, 0]
    fi = df.attrs.get("feature_importances", {})
    if fi:
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        names, values = zip(*sorted_fi)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
        bars = ax3.barh(list(names)[::-1], list(values)[::-1],
                        color=colors[::-1], alpha=0.85)
        ax3.set_xlabel("Importance", fontsize=10)
        ax3.set_title("Gradient Boosting Feature Importance", fontweight="bold")
        ax3.grid(True, axis="x", linestyle="--", alpha=0.5)
        ax3.tick_params(axis="both", labelsize=8.5)

    # ---- 4. Fuel efficiency (nm / kg) ------------------------------------
    ax4 = axes[1, 1]

    # Only show efficiency during airborne cruise/climb segments.
    # Mask out: ground taxi (IAS < AIRBORNE_IAS_THRESHOLD_KTS) and idle descent
    # (VS < IDLE_DESCENT_VS_THRESHOLD_FPM) to avoid misleading spikes from
    # near-idle fuel flow at low altitudes.
    eff = df["fuel_efficiency_nm_per_kg"].copy()
    ground_or_slow = df["ias_kts"] < AIRBORNE_IAS_THRESHOLD_KTS
    idle_descent = df["vertical_speed_fpm"] < IDLE_DESCENT_VS_THRESHOLD_FPM
    eff[ground_or_slow | idle_descent] = np.nan

    ax4.plot(t, eff, color="#6a1b9a", linewidth=2, zorder=3)
    ax4.fill_between(t, eff, alpha=0.15, color="#6a1b9a")

    # annotate cruise average
    cruise_eff = eff.loc[CRUISE_START_MIN:CRUISE_END_MIN]
    if cruise_eff.notna().any():
        mean_eff = cruise_eff.mean()
        ax4.axhline(mean_eff, color="#e65100", linewidth=1.2, linestyle="--")
        ax4.annotate(
            f"Cruise avg: {mean_eff:.3f} nm/kg",
            xy=(35, mean_eff),
            xytext=(0, 6), textcoords="offset points",
            fontsize=9, color="#e65100",
        )

    ax4.set_xlabel("Elapsed time (min)", fontsize=10)
    ax4.set_ylabel("Fuel Efficiency (nm / kg)", fontsize=10)
    ax4.set_title("Instantaneous Fuel Efficiency", fontweight="bold")
    ax4.set_xlim(t[0], t[-1])
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.tick_params(axis="both", labelsize=9)

    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Fuel estimation plot saved → %s", out_path)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shade_gaps(ax: plt.Axes, gap_minutes: list[int], t_max: float) -> None:
    """Shade the time windows where sensor data was unavailable."""
    # identify contiguous runs
    runs: list[tuple[int, int]] = []
    if not gap_minutes:
        return
    start = gap_minutes[0]
    prev = gap_minutes[0]
    for m in gap_minutes[1:]:
        if m != prev + 1:
            runs.append((start, prev))
            start = m
        prev = m
    runs.append((start, prev))

    for (s, e) in runs:
        ax.axvspan(s, e + 1, color="#ffccbc", alpha=0.55, zorder=1,
                   label="Sensor gap" if (s, e) == runs[0] else "_nolegend_")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)


def fuel_summary(df: pd.DataFrame) -> dict:
    """
    Return a dictionary of fuel metrics for the flight.

    Parameters
    ----------
    df : pd.DataFrame
        Augmented DataFrame from :func:`estimate_fuel_flow`.

    Returns
    -------
    dict
        Keys: total_estimated_fuel_kg, total_distance_nm,
              avg_efficiency_nm_per_kg, gap_minutes_filled.
    """
    # integrate fuel flow over time (trapezoidal rule, 1-min intervals)
    ff = df["fuel_flow_estimated_kgh"].to_numpy(dtype=float)
    dt_hr = 1.0 / 60.0
    total_fuel_kg = float(np.nansum(ff * dt_hr))

    # approximate distance from TAS
    tas = df["tas_kts"].to_numpy(dtype=float)
    airborne = df["altitude_ft"] > 1000
    total_dist_nm = float(np.nansum(tas[airborne] * dt_hr))

    eff = df["fuel_efficiency_nm_per_kg"]
    cruise_eff = eff.loc[CRUISE_START_MIN:CRUISE_END_MIN]
    avg_eff = float(cruise_eff.mean()) if cruise_eff.notna().any() else float("nan")

    return {
        "total_estimated_fuel_kg": round(total_fuel_kg, 1),
        "total_distance_nm": round(total_dist_nm, 1),
        "avg_cruise_efficiency_nm_per_kg": round(avg_eff, 4),
        "gap_minutes_filled": df.attrs.get("n_gap", 0),
    }
