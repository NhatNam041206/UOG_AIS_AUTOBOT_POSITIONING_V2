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
