from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Callable

from .config import EnvHelper
from .models import BaseEstimator, LinearEstimator, RandomForestEstimator, XGBoostEstimator
from .simulation import RunRecord, MockPhysicsEnv, read_run_history, write_run_history
from .strategies import DirectStrategy, EstimationStrategy, ResidualStrategy
from .view import TerminalDashboard


@dataclass
class Competitor:
    name: str
    strategy: EstimationStrategy
    estimator: BaseEstimator
    disqualified: bool = False
    real_errors: list[float] = field(default_factory=list)


class TournamentManager:
    """Coordinates simulation, training, evaluation, and champion selection."""

    def __init__(
        self,
        env: EnvHelper,
        dashboard: TerminalDashboard | None = None,
        estimator_factories: dict[str, Callable[[], BaseEstimator]] | None = None,
    ) -> None:
        self.env = env
        self.dashboard = dashboard or TerminalDashboard()
        self.sim_weight = env.get_val("SIM_WEIGHT", float, required=True)
        self.real_weight = env.get_val("REAL_WEIGHT", float, required=True)
        self.eval_interval = env.get_val("EVAL_INTERVAL", int, default=5)
        self.run_history_path = Path(env.get_val("RUN_HISTORY_PATH", str, default="run_history.csv"))
        self.system_log_path = Path(env.get_val("SYSTEM_LOG_PATH", str, default="system.log"))
        self.champion_model_path = Path(env.get_val("CHAMPION_MODEL_PATH", str, default="champion_model.pkl"))
        self.physics_env = MockPhysicsEnv(env)
        self._log_handler: logging.Handler | None = None
        self.logger = self._build_logger()
        self._rotation_index = 0
        self.competitors = self._build_competitors(estimator_factories)
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

    def _build_competitors(self, estimator_factories: dict[str, Callable[[], BaseEstimator]] | None) -> dict[str, Competitor]:
        factories = estimator_factories or {
            "linear": LinearEstimator,
            "random_forest": RandomForestEstimator,
            "xgboost": XGBoostEstimator,
        }
        selected_models = self.env.get_list("MODELS_TO_USE", default=factories.keys())
        competitors: dict[str, Competitor] = {}
        for model_key in selected_models:
            if model_key not in factories:
                continue
            for strategy in (DirectStrategy(), ResidualStrategy()):
                estimator = factories[model_key]()
                name = f"{strategy.short_name}_{estimator.short_name}"
                competitors[name] = Competitor(name=name, strategy=strategy, estimator=estimator)
        if not competitors:
            raise ValueError("MODELS_TO_USE did not produce any valid competitors")
        return competitors

    def run_experiment(self) -> str:
        run_count = self.env.get_val("NUM_SIM_RUNS", int, default=100)
        simulated_runs = self.physics_env.generate_runs(run_count, is_simulated=True)
        self._append_runs(simulated_runs)
        history = self.load_history()
        self._reset_disqualifications()
        self.train_all(history, use_grid_search=True)
        champion = self.elect_champion(history)
        self.logger.info("Experimental phase completed with champion %s", champion)
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
            active_model = self._choose_active_model(predictions)
            run.active_model = active_model
            self._append_runs([run])
            history.append(run)
            outputs.append(
                self.dashboard.render(
                    champion=self.champion_name,
                    battery_level=run.battery_level,
                    active_model=active_model,
                    predictions=predictions,
                )
            )
            self.logger.info(
                "Run %s recorded (simulated=%s, active_model=%s, actual_time=%.4f)",
                run.run_id,
                run.is_simulated,
                active_model,
                run.actual_time,
            )
            if index % self.eval_interval == 0:
                self._reset_disqualifications()
                self.train_all(history, use_grid_search=False)
                self.elect_champion(history)
        return outputs

    def train_all(self, history: list[RunRecord], use_grid_search: bool) -> None:
        for competitor in self.competitors.values():
            X = [record.features() for record in history]
            y = [competitor.strategy.training_target(record) for record in history]
            weights = [self.real_weight if not record.is_simulated else self.sim_weight for record in history]
            try:
                competitor.estimator.train(X, y, weights, use_grid_search=use_grid_search)
            except Exception as exc:  # pragma: no cover - exercised through integration behavior
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
                competitor.real_errors.append(abs(predicted_time - run.actual_time))
                predictions[competitor.name] = predicted_time
            except Exception as exc:
                competitor.disqualified = True
                self.logger.exception("Disqualifying %s during prediction: %s", competitor.name, exc)
        return predictions

    def elect_champion(self, history: list[RunRecord] | None = None) -> str:
        scores: dict[str, float] = {}
        for competitor in self.competitors.values():
            if competitor.disqualified:
                continue
            if competitor.real_errors:
                scores[competitor.name] = fmean(competitor.real_errors)
                continue
            if history:
                errors = self._score_history(competitor, history)
                if errors:
                    scores[competitor.name] = fmean(errors)
        if not scores:
            raise RuntimeError("No qualified models available to elect a champion")
        champion = min(scores, key=lambda name: scores[name])
        if champion != self.champion_name:
            previous = self.champion_name or "None"
            self.logger.info("Switching from %s to %s due to lower MAE", previous, champion)
        self.champion_name = champion
        self._save_champion(scores[champion])
        return champion

    def load_history(self) -> list[RunRecord]:
        return read_run_history(self.run_history_path)

    def _score_history(self, competitor: Competitor, history: list[RunRecord]) -> list[float]:
        errors: list[float] = []
        for record in history:
            try:
                raw_prediction = competitor.estimator.predict([record.features()])[0]
                predicted_time = competitor.strategy.prediction_to_time(record, raw_prediction)
                errors.append(abs(predicted_time - record.actual_time))
            except Exception as exc:
                competitor.disqualified = True
                self.logger.exception("Disqualifying %s during scoring: %s", competitor.name, exc)
                return []
        return errors

    def _append_runs(self, runs: list[RunRecord]) -> None:
        existing_runs = self.load_history()
        next_run_id = (existing_runs[-1].run_id or len(existing_runs)) + 1 if existing_runs else 1
        for run in runs:
            if run.run_id is None:
                run.run_id = next_run_id
                next_run_id += 1
        write_run_history(self.run_history_path, runs)

    def _choose_active_model(self, predictions: dict[str, float]) -> str:
        eligible_models = sorted(predictions)
        if not eligible_models:
            raise RuntimeError("No eligible models available")
        chosen = eligible_models[self._rotation_index % len(eligible_models)]
        self._rotation_index += 1
        if self.champion_name not in eligible_models:
            self.champion_name = chosen
        return chosen

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
