# Developer Documentation

This folder explains how the `UOG_AIS_AUTOBOT_POSITIONING_V2` codebase is organized and how the main pieces work together.

## What this project does

This project estimates robot travel time using a mix of:

- a **digital twin** simulation phase for initial training
- a **production** phase for real-world adaptation
- a **tournament manager** that compares multiple models and keeps the best performer as the current champion

The system follows a lightweight MVC-style structure:

- **Model layer**: machine-learning estimators and estimation strategies
- **View layer**: terminal dashboard output
- **Controller layer**: tournament flow, training, evaluation, and champion selection
- **Config layer**: `.env`-driven runtime settings

## High-level request flow

1. The app starts in `robot_positioning/app.py`.
2. `EnvHelper` loads configuration from `.env`.
3. `TournamentManager` builds the simulation environment, model pool, logging, and persistence paths.
4. In **experiment mode**:
   - simulated runs are generated
   - all enabled models are trained
   - the best model is elected as champion
5. In **production mode**:
   - historical runs are loaded
   - models are retrained with weighted simulated vs real data
   - each new run is scored by all active models
   - the current champion is re-evaluated every configured interval
6. `TerminalDashboard` prints the current champion, active model, battery state, and shadow predictions.

## Main concepts

### 1. Estimator
An estimator is the ML algorithm itself, such as linear regression, random forest, or XGBoost.

### 2. Strategy
A strategy decides what the model should learn:

- **Direct strategy**: predict the final travel time directly
- **Residual strategy**: predict only the difference between the physics baseline and the actual outcome

### 3. Competitor
A competitor is one strategy + one estimator.

Examples:

- `Direct_LR`
- `Residual_RF`
- `Residual_XGB`

### 4. Champion
The champion is the best currently qualified competitor based on MAE. It is persisted so the production phase can resume from the last known winner.

## Runtime files

These files are created when the app runs:

- `run_history.csv`: all simulated and real runs
- `system.log`: run events, errors, and champion switches
- `champion_model.pkl`: saved champion metadata

## Configuration

The main runtime settings live in `.env`.

Important examples:

- `APP_MODE`: `EXPERIMENT` or `PRODUCTION`
- `SIM_WEIGHT`: training weight for simulated data
- `REAL_WEIGHT`: training weight for real-world data
- `MODELS_TO_USE`: which estimators to include in the tournament
- `EVAL_INTERVAL`: how often to re-elect the champion in production

See `.env.example` for the full list.

## Discovery Analysis Notebook

`discovery_analysis.ipynb` is a self-contained Jupyter Notebook for offline analysis of
the robot's run history and evaluation of the Tournament Model.

### What it does

1. **Data generation** — if `run_history.csv` is missing, it generates a synthetic dataset
   (200 simulated + 50 real-world runs) using the built-in `MockPhysicsEnv`.
2. **Data loading & cleaning** — loads the CSV with full schema validation and applies the
   Transfer Learning weights (0.2 for simulated, 1.0 for real).
3. **Shadow prediction parsing** — expands `shadow_predictions_json` into a flat comparison
   table so every model's predictions can be audited per run.
4. **Exploratory Data Analysis** — three visualisation sections:
   - *Physics Check*: scatter of actual time vs. battery voltage to confirm the `V^1.2` power curve.
   - *Debt Analysis*: correlation heatmap + scatter showing how `prev_residual_angle` from a
     turn propagates into `total_calib_time` and `error_cm_or_deg` of the next forward segment.
   - *Error Distribution*: histogram of `error_cm_or_deg` per command type to detect systematic
     overshoot or undershoot bias.
5. **Tournament Training** — trains all 6 competitors (Direct × Residual strategies for
   LinearRegression, RandomForest, XGBoost) with Transfer Learning sample weights, evaluates
   each on the real-world hold-out set, and ranks them by MAE.
6. **Champion deep-dive** — actual-vs-predicted scatter, residuals histogram, and horizontal
   feature importance chart for the `Residual_RF` model.
7. **Model export** — saves the champion model as `champion_model.pkl` with all metadata
   needed for production inference.

### How to run

1. Install dependencies (includes `notebook`, `pandas`, `matplotlib`, `seaborn`):
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Launch Jupyter:
   ```bash
   jupyter notebook discovery_analysis.ipynb
   ```
   Or, to run non-interactively from the command line:
   ```bash
   jupyter nbconvert --to notebook --execute discovery_analysis.ipynb --output discovery_analysis_executed.ipynb
   ```
3. **Cell by cell or Run All** — the notebook is designed to run top-to-bottom.
   - If `run_history.csv` already exists in the repo root it will be loaded directly.
   - If it does not exist it will be generated automatically in the first two cells.

### Generated output files

| File | Description |
|---|---|
| `physics_check.png` | Battery voltage vs. actual time scatter (saved to repo root) |
| `debt_analysis.png` | Residual angle correlation heatmap and scatter |
| `error_distribution.png` | Error histograms per command type |
| `leaderboard.png` | Champion leaderboard bar chart |
| `residual_rf_plot.png` | Actual vs. predicted time for Residual_RF |
| `feature_importance.png` | Horizontal feature importance for Residual_RF |
| `champion_model.pkl` | Serialised champion model for production deployment |

### Customising the physics constants

If you have tuned `.env` parameters (e.g. `PHYSICS_SPEED`, `BATTERY_DECAY_EXPONENT`),
update the matching constants in **Section 4** of the notebook (`PHYSICS_SPEED`,
`TURN_SPEED_DEG_PER_SEC`) to keep the Residual strategy's physics baseline in sync.

## Testing

Install dependencies first:

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
python -m unittest discover -s tests -v
```

## Suggested reading order for new developers

1. `README.md`
2. `docs/codebase-files.md`
3. `robot_positioning/app.py`
4. `robot_positioning/controller.py`
5. `robot_positioning/simulation.py`
6. `robot_positioning/models.py`
7. `robot_positioning/strategies.py`
8. `tests/test_robot_positioning.py`
