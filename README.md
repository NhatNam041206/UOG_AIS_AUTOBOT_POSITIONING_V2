# UOG_AIS_AUTOBOT_POSITIONING_V2

Python MVC/OOP robot endpoint estimation engine with a tournament of AI models for route-time prediction.

## Setup

1. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and tune the physics/config values.
3. Run experiment mode:
   ```bash
   python -m robot_positioning --env-file .env
   ```
4. Switch `APP_MODE=PRODUCTION` to run real-world adaptation and champion re-evaluation.

## Core Engine Highlights

- **MockPhysicsEnv** uses:
  - `T_actual = [(Tiles × 1.2) + (Turns × 1.8)] × (12.0 / V_start)^1.2 + Noise`
- **6-model tournament**:
  - `[LinearRegression, RandomForest, XGBoost] × [Direct, Residual]`
- **Champion election**:
  - every `EVAL_INTERVAL` runs (default 10)
  - lowest MAE on **real-world data only**
- **Transfer learning**:
  - weighted fit (`SIM_WEIGHT=0.2`, `REAL_WEIGHT=1.0` by default)
  - retraining after each production run
- **Time distribution helper**:
  - splits predicted total time into forward vs turning command budgets

## Notebook

Open `robot_discovery.ipynb` for EDA and tournament visualization:

```bash
jupyter notebook robot_discovery.ipynb
```

## Testing

```bash
python -m unittest discover -s tests -v
```
