from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from robot_positioning.config import EnvHelper
from robot_positioning.controller import TournamentManager


def write_env(path: Path, mode: str = "EXPERIMENT", eval_interval: int = 3) -> None:
    path.write_text(
        "\n".join(
            [
                f"APP_MODE={mode}",
                "SIM_WEIGHT=0.2",
                "REAL_WEIGHT=1.0",
                "MODELS_TO_USE=linear,random_forest,xgboost",
                "NUM_SIM_RUNS=12",
                f"EVAL_INTERVAL={eval_interval}",
                "PRODUCTION_RUNS=4",
                "RANDOM_SEED=7",
                "PHYSICS_SPEED=1.7",
                "BATTERY_DECAY_EXPONENT=1.2",
                "ANGULAR_DRIFT_FACTOR=0.08",
                "PAYLOAD_FACTOR=0.6",
                "NOISE_STD=0.02",
                "REAL_WORLD_BIAS=0.12",
                "DISTANCE_MIN=1.5",
                "DISTANCE_MAX=10.0",
                "BATTERY_MIN=0.45",
                "BATTERY_MAX=0.95",
                "HEADING_MIN=-0.4",
                "HEADING_MAX=0.4",
                "TERRAIN_MIN=0.9",
                "TERRAIN_MAX=1.2",
                "PAYLOAD_MIN=0.0",
                "PAYLOAD_MAX=1.0",
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
                champion = manager.run_experiment()
                self.assertTrue(champion.startswith(("Direct_", "Residual_")))
                history_after_experiment = manager.load_history()
                self.assertEqual(12, len(history_after_experiment))
                self.assertTrue(all(record.is_simulated for record in history_after_experiment))
                champion_payload = pickle.loads((temp_path / "champion_model.pkl").read_bytes())
                self.assertEqual(champion, champion_payload["champion_name"])

                outputs = manager.run_production(manager.physics_env.generate_runs(4, is_simulated=False))
                history_after_production = manager.load_history()
                self.assertEqual(16, len(history_after_production))
                self.assertEqual(4, sum(not record.is_simulated for record in history_after_production))
                self.assertTrue(outputs)
                self.assertIn("Champion:", outputs[-1])
                self.assertIn("Shadow Predictions:", outputs[-1])
                self.assertIn("Switching from", (temp_path / "system.log").read_text(encoding="utf-8"))
            finally:
                manager.close()

    def test_prediction_failure_disqualifies_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            env_path = temp_path / ".env"
            write_env(env_path, eval_interval=10)
            env = EnvHelper(env_path)
            manager = TournamentManager(env)
            try:
                manager.run_experiment()
                manager.competitors["Direct_LR"].estimator.predict = lambda X: (_ for _ in ()).throw(RuntimeError("boom"))

                outputs = manager.run_production(manager.physics_env.generate_runs(1, is_simulated=False))

                self.assertTrue(outputs)
                self.assertTrue(manager.competitors["Direct_LR"].disqualified)
                self.assertIn("Disqualifying Direct_LR during prediction", (temp_path / "system.log").read_text(encoding="utf-8"))
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
