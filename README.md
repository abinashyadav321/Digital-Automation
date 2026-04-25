# Digital-Automation — Aircraft Digital Twin

This project demonstrates a **Digital Twin** approach to aircraft performance
analysis.  Using telemetry data from **Nepal Airlines Flight RA205
(KTM → DEL)**, it provides a Python-based analytics pipeline that:

* **Visualises the complete flight profile** — altitude, airspeed (IAS / TAS /
  Mach), vertical speed, and ground track.
* **Estimates fuel flow** where the physical fuel-flow sensor was unavailable
  (NaN gaps) using both a physics-inspired linear model and a Gradient Boosting
  regressor trained on the known measurements.
* **Reports fuel efficiency** (nm / kg) across all flight phases.

---

## Aircraft & Route

| Field            | Value                                          |
|------------------|------------------------------------------------|
| Flight           | Nepal Airlines RA205                           |
| Route            | Kathmandu (KTM / VNKT) → Delhi (DEL / VIDP)   |
| Aircraft         | Airbus A319-100                                |
| Date             | 15 March 2024                                  |
| Block time       | ~90 minutes                                    |
| Cruise altitude  | FL350 (35 000 ft)                              |
| Cruise Mach      | ~0.78                                          |

---

## Project Structure

```
Digital-Automation/
├── data/
│   └── ra205_telemetry.csv     # 91-row telemetry (1-min sampling, NaN gaps)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load, validate, and characterise telemetry
│   ├── flight_profile.py       # Flight-profile & engine-performance plots
│   └── fuel_estimator.py       # Physics + ML fuel-flow gap-filling
├── output/                     # Generated PNG charts (git-ignored)
├── main.py                     # Pipeline orchestrator / CLI entry-point
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py

# Optional: specify custom paths
python main.py --data data/ra205_telemetry.csv --output output/
```

Charts are written to `output/`:

| File                      | Contents                                        |
|---------------------------|-------------------------------------------------|
| `flight_profile.png`      | Altitude · airspeed · vertical speed · ground track |
| `engine_performance.png`  | Engine 1 & 2 N1 fan speed over time             |
| `fuel_estimation.png`     | Measured vs estimated fuel flow · residuals · feature importance · efficiency |

---

## Sensor Gap Simulation

Two windows of `fuel_flow_total_kgh` are set to `NaN` in the telemetry CSV
to simulate realistic sensor dropouts:

* **t = 32 – 44 min** (13 min during cruise) — left-engine FMC fuel-flow sensor failure.
* **t = 70 – 72 min** (3 min during descent) — brief transient dropout.

The pipeline trains both models exclusively on the **non-NaN rows** and then
predicts the missing values, demonstrating a Digital Twin's ability to
maintain situational awareness despite partial sensor loss.

---

## Fuel Estimation Models

### Physics-based model
A linear regression on physics-inspired features (N1², TAS², altitude,
Mach, OAT) fitted on the known rows.  Provides an interpretable baseline.

### Gradient Boosting regressor (GBR)
A `GradientBoostingRegressor` (200 estimators, depth 4) that captures
non-linearities across flight phases.  Used to fill the actual NaN gaps.

---

## Key Outputs (example run)

```
Total estimated fuel burned : 4 730 kg
Estimated flight distance   : 443 nm
Avg cruise efficiency       : 0.1852 nm/kg
Sensor-gap minutes filled   : 16
```

---

## Dependencies

| Package       | Purpose                              |
|---------------|--------------------------------------|
| pandas        | Telemetry DataFrame manipulation     |
| numpy         | Numerical computations               |
| matplotlib    | All visualisations                   |
| scikit-learn  | Gradient Boosting & Linear Regression|
| scipy         | Standard atmosphere utilities        |

