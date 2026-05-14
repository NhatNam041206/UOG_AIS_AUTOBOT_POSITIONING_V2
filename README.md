# UOG_AIS_AUTOBOT_POSITIONING_V2

Python MVC/OOP robot positioning estimation engine with a digital-twin experiment phase and a transfer-learning production phase.

## Setup

1. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and tune the physics/config values.
3. Run the experiment phase:
   ```bash
   python -m robot_positioning --env-file .env
   ```
4. Switch `APP_MODE=PRODUCTION` and run again to execute interleaved production evaluation.

## Documentation

- `docs/README.md`: high-level developer guide
- `docs/codebase-files.md`: file-by-file codebase reference

## Key outputs

- `run_history.csv`: saved run history with the `is_simulated` flag.
- `system.log`: run/error/champion-switch logging.
- `champion_model.pkl`: persisted champion metadata.
