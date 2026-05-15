from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from robot_positioning.config import EnvHelper
from robot_positioning.controller import TournamentManager
from robot_positioning.simulation import RunRecord


def write_env(path: Path, mode: str = "EXPERIMENT", eval_interval: int = 10) -> None:
    path.write_text(
        "\n".join(
            [
                f"APP_MODE={mode}",
                "SIM_WEIGHT=0.2",
                "REAL_WEIGHT=1.0",
                "MODELS_TO_USE=linear,random_forest,xgboost",
                "NUM_SIM_RUNS=20",
                f"EVAL_INTERVAL={eval_interval}",
                "PRODUCTION_RUNS=4",
                "RANDOM_SEED=7",
                "PHYSICS_TILE_TIME=1.2",
                "PHYSICS_TURN_TIME=1.8",
                "PHYSICS_NOMINAL_VOLTAGE=12.0",
                "BATTERY_DECAY_EXPONENT=1.2",
                "TIME_NOISE_STD=0.2",
                "TIME_NOISE_STD_REAL=0.4",
                "TOTAL_TILES_MIN=20",
                "TOTAL_TILES_MAX=30",
                "NUM_CORNERS_MIN=4",
                "NUM_CORNERS_MAX=8",
                "START_BATTERY_MIN=10.0",
                "START_BATTERY_MAX=12.0",
                "CALIBRATIONS_MIN=1",
                "CALIBRATIONS_MAX=10",
                "ENDPOINT_ERROR_ABS_MAX_CM=50.0",
                "ENDPOINT_DEVIATION_ABS_MAX_DEG=15.0",
                "REAL_WORLD_ERROR_MULTIPLIER=1.2",
                "LR_FIT_INTERCEPT_OPTIONS=true,false",
                "RF_N_ESTIMATORS_OPTIONS=20,30",
                "RF_MAX_DEPTH_OPTIONS=3,4",
                "XGB_MAX_DEPTH_OPTIONS=3,4",
                "XGB_LEARNING_RATE_OPTIONS=0.05,0.1",
                "XGB_N_ESTIMATORS_OPTIONS=20,30",
                f"RUN_HISTORY_PATH={path.parent / 'run_history.csv'}",
                f"SYSTEM_LOG_PATH={path.parent / 'system.log'}",
                f"CHAMPION_MODEL_PATH={path.parent / 'champion_model.pkl'}",
            ]
        ),
        encoding="utf-8",
    )


class TournamentManagerTests(unittest.TestCase):
    def test_experiment_and_production_persist_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            env_path = temp_path / ".env"
            write_env(env_path)
            env = EnvHelper(env_path)
            manager = TournamentManager(env)
            try:
                self.assertEqual(6, len(manager.competitors))
                champion = manager.run_experiment()
                self.assertTrue(champion.startswith(("Direct_", "Residual_")))

                history_after_experiment = manager.load_history()
                self.assertEqual(20, len(history_after_experiment))
                self.assertTrue(all(record.is_simulated for record in history_after_experiment))
                champion_payload = pickle.loads((temp_path / "champion_model.pkl").read_bytes())
                self.assertEqual(champion, champion_payload["champion_name"])

                outputs = manager.run_production(manager.physics_env.generate_runs(4, is_simulated=False))
                history_after_production = manager.load_history()
                self.assertEqual(24, len(history_after_production))
                self.assertEqual(4, sum(not record.is_simulated for record in history_after_production))
                self.assertTrue(outputs)
                self.assertIn("Champion:", outputs[-1])
                self.assertIn("Shadow Predictions:", outputs[-1])
                self.assertIn("Time Distribution:", outputs[-1])
            finally:
                manager.close()

    def test_physics_formula_matches_when_noise_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            env_path = temp_path / ".env"
            write_env(env_path)
            content = env_path.read_text(encoding="utf-8")
            content = content.replace("TIME_NOISE_STD=0.2", "TIME_NOISE_STD=0.0")
            content = content.replace("TIME_NOISE_STD_REAL=0.4", "TIME_NOISE_STD_REAL=0.0")
            env_path.write_text(content, encoding="utf-8")

            env = EnvHelper(env_path)
            manager = TournamentManager(env)
            try:
                run = manager.physics_env.generate_run(is_simulated=True)
                baseline = (run.total_tiles * manager.physics_env.tile_time) + (run.num_corners * manager.physics_env.turn_time)
                expected = baseline * ((manager.physics_env.nominal_voltage / run.start_battery_v) ** manager.physics_env.battery_exponent)
                self.assertAlmostEqual(expected, run.actual_time_total, places=9)
            finally:
                manager.close()

    def test_retrain_after_each_production_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            env_path = temp_path / ".env"
            write_env(env_path, eval_interval=10)
            env = EnvHelper(env_path)
            manager = TournamentManager(env)
            try:
                manager.run_experiment()
                counter = {"calls": 0}
                original_train_all = manager.train_all

                def wrapped(history, use_grid_search):
                    counter["calls"] += 1
                    return original_train_all(history, use_grid_search)

                manager.train_all = wrapped  # type: ignore[method-assign]
                manager.run_production(manager.physics_env.generate_runs(3, is_simulated=False))
                self.assertEqual(4, counter["calls"])  # initial training + 3 per-run retrains
            finally:
                manager.close()

    def test_champion_election_uses_real_world_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            env_path = temp_path / ".env"
            write_env(env_path)
            env = EnvHelper(env_path)
            manager = TournamentManager(env)
            try:
                history = [
                    RunRecord(total_tiles=20, num_corners=4, start_battery_v=12.0, calibrations_total=1, actual_time_total=40.0, is_simulated=True),
                    RunRecord(total_tiles=21, num_corners=5, start_battery_v=11.5, calibrations_total=2, actual_time_total=42.0, is_simulated=True),
                    RunRecord(total_tiles=22, num_corners=6, start_battery_v=11.0, calibrations_total=3, actual_time_total=45.0, is_simulated=False),
                    RunRecord(total_tiles=23, num_corners=7, start_battery_v=10.8, calibrations_total=4, actual_time_total=48.0, is_simulated=False),
                ]

                captured = {"all_real": True}

                def fake_score(_competitor, records):
                    captured["all_real"] = captured["all_real"] and all(not r.is_simulated for r in records)
                    return [1.0]

                manager._score_history = fake_score  # type: ignore[method-assign]
                champion = manager.elect_champion(history)
                self.assertTrue(champion)
                self.assertTrue(captured["all_real"])
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
