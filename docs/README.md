# Developer Documentation

## Architecture

The project follows lightweight MVC and OOP design:

- **Model**: estimator classes + direct/residual strategies
- **View**: terminal dashboard rendering
- **Controller**: tournament lifecycle, retraining, champion election
- **Config**: `.env`-driven runtime and hyperparameter values

## Engine Scope

Each run uses the schema:

- `run_id`
- `total_tiles` (20-60)
- `num_corners` (4-8)
- `start_battery_v` (10.0-12.0)
- `calibrations_total` (1-10)
- `actual_time_total` (target)
- `endpoint_error_cm` (±50)
- `endpoint_deviated_deg` (±15)
- `is_simulated`

`MockPhysicsEnv` computes runtime with:

`T_actual = [(Tiles × tile_time) + (Turns × turn_time)] × (nominal_voltage / V_start)^battery_exponent + noise`

(`tile_time`, `turn_time`, `nominal_voltage`, `battery_exponent`, and noise params are loaded from `.env`.)

## Tournament

Pool size is 6 competitors:

- Estimators: `LinearRegression`, `RandomForest`, `XGBoost`
- Strategies: `Direct`, `Residual`

Champion selection:

- Runs every `EVAL_INTERVAL` (default 10)
- Metric: lowest MAE on **real-world records only**

Transfer-learning fit weights:

- Simulated data: `SIM_WEIGHT` (default 0.2)
- Real-world data: `REAL_WEIGHT` (default 1.0)

Models are retrained after each production run.

## Notebook

Use `robot_discovery.ipynb` for:

1. battery vs total-time profiling
2. endpoint error vs calibration analysis
3. 6-model MAE leaderboard + champion highlighting
4. champion feature-importance plot
5. simulated→real learning-curve visualization

## Tests

```bash
python -m unittest discover -s tests -v
```
