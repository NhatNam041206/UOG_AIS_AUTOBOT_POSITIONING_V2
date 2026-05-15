from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable

from .config import EnvHelper
from .models import BaseEstimator, LinearEstimator, RandomForestEstimator, XGBoostEstimator
from .simulation import MockPhysicsEnv, RunRecord, read_run_history, write_run_history
from .strategies import DirectStrategy, EstimationStrategy, ResidualStrategy
from .view import TerminalDashboard

MIN_BASELINE_TIME = 1e-6


@dataclass
class Competitor:
    name: str
    strategy: EstimationStrategy
    estimator: BaseEstimator
    disqualified: bool = False


class TournamentManager:
    """Coordinates training, prediction, and champion election for robot endpoint estimation."""

    def __init__(self, env: EnvHelper, dashboard: TerminalDashboard | None = None) -> None:
        self.env = env
        self.dashboard = dashboard or TerminalDashboard()
        self.sim_weight = env.get_val("SIM_WEIGHT", float, required=True)
        self.real_weight = env.get_val("REAL_WEIGHT", float, required=True)
        self.eval_interval = env.get_val("EVAL_INTERVAL", int, default=10)
        self.run_history_path = Path(env.get_val("RUN_HISTORY_PATH", str, default="run_history.csv"))
        self.system_log_path = Path(env.get_val("SYSTEM_LOG_PATH", str, default="system.log"))
        self.champion_model_path = Path(env.get_val("CHAMPION_MODEL_PATH", str, default="champion_model.pkl"))

        self.physics_env = MockPhysicsEnv(env)
        self._log_handler: logging.Handler | None = None
        self.logger = self._build_logger()
        self.competitors = self._build_competitors()
        self.champion_name = ""

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"robot_positioning.{self.system_log_path}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            self.system_log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(self.system_log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
            self._log_handler = handler
        return logger

    def close(self) -> None:
        if self._log_handler is None:
            return
        self.logger.removeHandler(self._log_handler)
        self._log_handler.close()
        self._log_handler = None

    def _parse_int_options(self, key: str, default: list[int]) -> list[int]:
        values = self.env.get_list(key, default=[str(item) for item in default])
        return [int(value) for value in values]

    def _parse_float_options(self, key: str, default: list[float]) -> list[float]:
        values = self.env.get_list(key, default=[str(item) for item in default])
        return [float(value) for value in values]

    def _parse_bool_options(self, key: str, default: list[bool]) -> list[bool]:
        default_values = ["true" if value else "false" for value in default]
        values = self.env.get_list(key, default=default_values)
        return [value.strip().lower() in {"1", "true", "yes", "on"} for value in values]

    def _build_competitors(self) -> dict[str, Competitor]:
        lr_grid = {"fit_intercept": self._parse_bool_options("LR_FIT_INTERCEPT_OPTIONS", [True, False])}
        rf_grid = {
            "n_estimators": self._parse_int_options("RF_N_ESTIMATORS_OPTIONS", [80, 120]),
            "max_depth": self._parse_int_options("RF_MAX_DEPTH_OPTIONS", [4, 6]),
        }
        xgb_grid = {
            "max_depth": self._parse_int_options("XGB_MAX_DEPTH_OPTIONS", [3, 5]),
            "learning_rate": self._parse_float_options("XGB_LEARNING_RATE_OPTIONS", [0.05, 0.1]),
            "n_estimators": self._parse_int_options("XGB_N_ESTIMATORS_OPTIONS", [60, 100]),
        }

        factories: dict[str, Callable[[], BaseEstimator]] = {
            "linear": lambda: LinearEstimator(lr_grid),
            "random_forest": lambda: RandomForestEstimator(rf_grid),
            "xgboost": lambda: XGBoostEstimator(xgb_grid),
        }
        selected_models = self.env.get_list("MODELS_TO_USE", default=factories.keys())

        competitors: dict[str, Competitor] = {}
        for model_key in selected_models:
            if model_key not in factories:
                continue
            for strategy in (
                DirectStrategy(),
                ResidualStrategy(tile_time=self.physics_env.tile_time, turn_time=self.physics_env.turn_time),
            ):
                estimator = factories[model_key]()
                name = f"{strategy.short_name}_{estimator.short_name}"
                competitors[name] = Competitor(name=name, strategy=strategy, estimator=estimator)

        if len(competitors) != 6:
            raise ValueError("Tournament must include 6 competitors (3 estimators × 2 strategies)")
        return competitors

    def run_experiment(self) -> str:
        run_count = self.env.get_val("NUM_SIM_RUNS", int, default=100)
        simulated_runs = self.physics_env.generate_runs(run_count, is_simulated=True)
        self._append_runs(simulated_runs)
        history = self.load_history()
        self._reset_disqualifications()
        self.train_all(history, use_grid_search=True)
        champion = self.elect_champion(history)
        self.logger.info("Experiment completed with champion %s", champion)
        return champion

    def run_production(self, actual_runs: list[RunRecord] | None = None) -> list[str]:
        history = self.load_history()
        if not history:
            raise RuntimeError("Production requires prior experimental history")

        self._load_saved_champion()
        self._reset_disqualifications()
        self.train_all(history, use_grid_search=False)
        if not self.champion_name:
            self.elect_champion(history)

        outputs: list[str] = []
        actual_runs = actual_runs or self.physics_env.generate_runs(
            self.env.get_val("PRODUCTION_RUNS", int, default=self.eval_interval),
            is_simulated=False,
        )

        for index, run in enumerate(actual_runs, start=1):
            predictions = self.predict_all(run)
            if not predictions:
                raise RuntimeError("All tournament models were disqualified")
            if self.champion_name not in predictions:
                self.champion_name = sorted(predictions)[0]

            champion_pred = predictions[self.champion_name]
            distribution = self.distribute_time(champion_pred, run)

            self._append_runs([run])
            history.append(run)
            outputs.append(
                self.dashboard.render(
                    champion=self.champion_name,
                    start_battery_v=run.start_battery_v,
                    active_model=self.champion_name,
                    predictions=predictions,
                    time_distribution=distribution,
                )
            )

            self._reset_disqualifications()
            self.train_all(history, use_grid_search=False)

            if index % self.eval_interval == 0:
                self.elect_champion(history)

        return outputs

    def train_all(self, history: list[RunRecord], use_grid_search: bool) -> None:
        X = [record.features() for record in history]
        weights = [self.real_weight if not record.is_simulated else self.sim_weight for record in history]
        for competitor in self.competitors.values():
            y = [competitor.strategy.training_target(record) for record in history]
            try:
                competitor.estimator.train(X, y, weights, use_grid_search=use_grid_search)
            except Exception as exc:  # pragma: no cover
                competitor.disqualified = True
                self.logger.exception("Disqualifying %s during training: %s", competitor.name, exc)

    def predict_all(self, run: RunRecord) -> dict[str, float]:
        predictions: dict[str, float] = {}
        for competitor in self.competitors.values():
            if competitor.disqualified:
                continue
            try:
                raw_prediction = competitor.estimator.predict([run.features()])[0]
                predicted_time = competitor.strategy.prediction_to_time(run, raw_prediction)
                predictions[competitor.name] = predicted_time
            except Exception as exc:
                competitor.disqualified = True
                self.logger.exception("Disqualifying %s during prediction: %s", competitor.name, exc)
        return predictions

    def elect_champion(self, history: list[RunRecord] | None = None) -> str:
        if history is None:
            history = self.load_history()

        real_history = [record for record in history if not record.is_simulated]
        score_history = real_history or history

        scores: dict[str, float] = {}
        for competitor in self.competitors.values():
            if competitor.disqualified:
                continue
            errors = self._score_history(competitor, score_history)
            if errors:
                scores[competitor.name] = fmean(errors)

        if not scores:
            raise RuntimeError("No qualified models available to elect a champion")

        champion = min(scores, key=lambda name: scores[name])
        if champion != self.champion_name:
            previous = self.champion_name or "None"
            self.logger.info("Switching from %s to %s due to lower real-world MAE", previous, champion)
        self.champion_name = champion
        self._save_champion(scores[champion])
        return champion

    def distribute_time(self, predicted_total_time: float, run: RunRecord) -> dict[str, float]:
        """Split total route time proportionally into forward and turning budgets."""
        forward_baseline = run.total_tiles * self.physics_env.tile_time
        turn_baseline = run.num_corners * self.physics_env.turn_time
        baseline_total = max(forward_baseline + turn_baseline, MIN_BASELINE_TIME)

        forward_share = forward_baseline / baseline_total
        turn_share = turn_baseline / baseline_total

        forward_time_total = predicted_total_time * forward_share
        turn_time_total = predicted_total_time * turn_share

        tile_count = run.total_tiles if run.total_tiles > 0 else 1
        corner_count = run.num_corners if run.num_corners > 0 else 1

        return {
            "forward_time_total": forward_time_total,
            "turn_time_total": turn_time_total,
            "forward_time_per_tile": forward_time_total / tile_count,
            "turn_time_per_corner": turn_time_total / corner_count,
        }

    def load_history(self) -> list[RunRecord]:
        return read_run_history(self.run_history_path)

    def export_leaderboard(self, history: list[RunRecord] | None = None) -> dict[str, float]:
        history = history or self.load_history()
        real_history = [record for record in history if not record.is_simulated]
        score_history = real_history or history
        leaderboard: dict[str, float] = {}
        for competitor in self.competitors.values():
            if competitor.disqualified:
                continue
            errors = self._score_history(competitor, score_history)
            if errors:
                leaderboard[competitor.name] = fmean(errors)
        return dict(sorted(leaderboard.items(), key=lambda item: item[1]))

    def _score_history(self, competitor: Competitor, history: list[RunRecord]) -> list[float]:
        errors: list[float] = []
        for record in history:
            try:
                raw_prediction = competitor.estimator.predict([record.features()])[0]
                predicted_time = competitor.strategy.prediction_to_time(record, raw_prediction)
                errors.append(abs(predicted_time - record.actual_time_total))
            except Exception as exc:
                competitor.disqualified = True
                self.logger.exception("Disqualifying %s during scoring: %s", competitor.name, exc)
                return []
        return errors

    def _append_runs(self, runs: list[RunRecord]) -> None:
        existing = self.load_history()
        next_run_id = (existing[-1].run_id or len(existing)) + 1 if existing else 1
        for run in runs:
            if run.run_id is None:
                run.run_id = next_run_id
                next_run_id += 1
        write_run_history(self.run_history_path, runs)

    def _reset_disqualifications(self) -> None:
        for competitor in self.competitors.values():
            competitor.disqualified = False

    def _save_champion(self, score: float) -> None:
        self.champion_model_path.parent.mkdir(parents=True, exist_ok=True)
        with self.champion_model_path.open("wb") as handle:
            pickle.dump({"champion_name": self.champion_name, "mae": score}, handle)

    def _load_saved_champion(self) -> None:
        if not self.champion_model_path.exists():
            return
        with self.champion_model_path.open("rb") as handle:
            payload = pickle.load(handle)
        self.champion_name = payload.get("champion_name", "")

    def serialize_shadow_predictions(self, run: RunRecord) -> str:
        predictions = self.predict_all(run)
        return json.dumps({name: float(value) for name, value in sorted(predictions.items())})
