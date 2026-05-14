# Codebase File Guide

This document explains the purpose of each important file in the repository.

## Repository root

### `/README.md`
Short project-level setup and usage guide. A new developer should start here for installation and the basic run flow.

### `/.env.example`
Template configuration file. It shows all expected environment variables, including model selection, physics constants, persistence paths, and production settings.

### `/requirements.txt`
Python dependencies required by the project. This must be installed before running the app or the tests.

### `/.gitignore`
Lists files that should not be committed, including Python cache files, local environments, and runtime artifacts like logs and generated model files.

### `/LICENSE`
Repository license file.

## Documentation

### `/docs/README.md`
High-level developer guide for understanding the system architecture and runtime flow.

### `/docs/codebase-files.md`
This file. It gives a file-by-file reference for the repository.

## Application package: `/robot_positioning`

### `/robot_positioning/__init__.py`
Package export file. It re-exports the main public entry points:

- `EnvHelper`
- `TournamentManager`
- `TerminalDashboard`
- `main`

This makes the package easier to import from other code.

### `/robot_positioning/__main__.py`
Allows the package to be run with:

```bash
python -m robot_positioning
```

It simply calls `main()` from `app.py`.

### `/robot_positioning/app.py`
The CLI entrypoint.

Responsibilities:

- parse command-line arguments
- load the `.env` file
- create the `TournamentManager`
- choose between experiment mode and production mode

This is the best place to look when you want to understand how execution begins.

### `/robot_positioning/config.py`
Contains `EnvHelper`, the configuration helper.

Responsibilities:

- read values from a `.env` file
- merge those values with real environment variables
- provide typed access through `get_val()`
- provide comma-separated list parsing through `get_list()`

Why it matters:

- keeps configuration out of business logic
- avoids hardcoded values in the rest of the application
- makes tests easier by supporting overrides

### `/robot_positioning/models.py`
Contains the estimator abstraction and concrete ML model wrappers.

Key classes:

- `BaseEstimator`: shared training and prediction interface
- `LinearEstimator`: linear regression model
- `RandomForestEstimator`: random forest regressor
- `XGBoostEstimator`: XGBoost regressor

Responsibilities:

- standardize `.train()` and `.predict()` across models
- optionally run `GridSearchCV` during the experiment phase
- keep model-specific parameter grids close to the model definitions

### `/robot_positioning/strategies.py`
Contains the estimation strategy layer.

Key classes:

- `EstimationStrategy`: abstract base class
- `DirectStrategy`: predicts actual travel time directly
- `ResidualStrategy`: predicts the correction on top of the physics baseline

Why it exists:

It separates **what to predict** from **which ML algorithm to use**. That lets the tournament manager combine each strategy with each estimator.

### `/robot_positioning/simulation.py`
Contains simulation and persistence logic.

Key parts:

- `RunRecord`: data object for one run
- `MockPhysicsEnv`: digital twin generator
- `write_run_history()`: append runs to CSV
- `read_run_history()`: load existing runs from CSV

Responsibilities:

- define the feature structure used across the codebase
- generate simulated runs with battery decay, drift, payload, terrain, and noise
- serialize and deserialize run history

This file is important because it is the bridge between the physical assumptions and the ML pipeline.

### `/robot_positioning/controller.py`
Contains the main orchestration logic.

Key types:

- `Competitor`: one strategy-estimator pairing plus tournament state
- `TournamentManager`: the main controller

Responsibilities of `TournamentManager`:

- build the model pool
- create the logger
- run the experiment phase
- run the production phase
- train all competitors with sample weights
- predict with all competitors
- disqualify models that fail during training or prediction
- compare MAE values and elect the champion
- save and reload champion metadata
- rotate active models during production for interleaved testing

If a new developer wants to understand the application's behavior, this is the most important file in the repository.

### `/robot_positioning/view.py`
Contains `TerminalDashboard`.

Responsibilities:

- format terminal-friendly output
- display the champion
- display the currently active model
- display battery level
- display shadow predictions from the non-champion models

This is intentionally simple so that presentation logic stays outside the controller.

## Tests

### `/tests/test_robot_positioning.py`
Integration-style unit tests for the core workflow.

What it checks:

- experiment mode creates a champion and persists artifacts
- production mode appends real runs and renders dashboard output
- a prediction failure disqualifies the broken model and logs the error

This test file is the best place to understand the expected end-to-end behavior.

## How the files connect

A simple mental model is:

- `app.py` starts the program
- `config.py` loads settings
- `controller.py` runs the workflow
- `simulation.py` provides data and persistence
- `models.py` provides learning algorithms
- `strategies.py` decides training targets
- `view.py` prints results
- `tests/test_robot_positioning.py` verifies the flow
