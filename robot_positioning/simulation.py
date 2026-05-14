from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import EnvHelper

CSV_FIELDNAMES = [
    "run_id",
    "segment_index",
    "is_simulated",
    "command_id",
    "target_units",
    "start_battery_v",
    "prev_cmd_id",
    "prev_residual_angle",
    "calibrations_count",
    "total_calib_time",
    "avg_drift_angle",
    "actual_time_consumed",
    "actual_dist_reached",
    "error_cm_or_deg",
    "active_champion_id",
    "shadow_predictions_json",
]
COMMAND_FORWARD = 0
COMMAND_TURN_LEFT = 1
COMMAND_TURN_RIGHT = 2
TURN_COMMAND_IDS = (COMMAND_TURN_LEFT, COMMAND_TURN_RIGHT)
MIN_EXPECTED_TIME_SECONDS = 0.1
MIN_FORWARD_SPEED_DIVISOR = 0.1
MIN_TURN_SPEED_DIVISOR = 1.0
PREV_RESIDUAL_DRIFT_FACTOR = 0.15
ADDITIONAL_RESIDUAL_ANGULAR_FACTOR = 0.05
FORWARD_ERROR_SCALE = 0.05
TURN_ERROR_SCALE = 0.08
PREV_RESIDUAL_GAUSSIAN_MEAN_FACTOR = 0.1
PREV_RESIDUAL_DECAY = 0.5
DRIFT_CONTRIBUTION = 0.3


@dataclass
class RunRecord:
    run_id: int | None = None
    segment_index: int = 1
    is_simulated: bool = True
    command_id: int = 0
    target_units: float = 0.0
    start_battery_v: float = 0.0
    prev_cmd_id: int = 0
    prev_residual_angle: float = 0.0
    calibrations_count: int = 0
    total_calib_time: float = 0.0
    avg_drift_angle: float = 0.0
    actual_time_consumed: float = 0.0
    actual_dist_reached: float = 0.0
    error_cm_or_deg: float = 0.0
    active_champion_id: str = ""
    shadow_predictions_json: str = "{}"
    forward_units_per_second: float = 1.7
    turn_degrees_per_second: float = 90.0

    def features(self) -> list[float]:
        command_battery_interaction = self.start_battery_v * float(self.command_id)
        turn_debt_interaction = self.prev_residual_angle * float(self.command_id)
        return [
            float(self.command_id),
            self.target_units,
            self.start_battery_v,
            float(self.prev_cmd_id),
            self.prev_residual_angle,
            float(self.calibrations_count),
            self.total_calib_time,
            self.avg_drift_angle,
            command_battery_interaction,
            turn_debt_interaction,
        ]

    def expected_time(self) -> float:
        if self.command_id == COMMAND_FORWARD:
            return max(MIN_EXPECTED_TIME_SECONDS, self.target_units / max(self.forward_units_per_second, MIN_FORWARD_SPEED_DIVISOR))
        return max(MIN_EXPECTED_TIME_SECONDS, self.target_units / max(self.turn_degrees_per_second, MIN_TURN_SPEED_DIVISOR))

    def to_csv_row(self) -> dict[str, str | float | int]:
        return {
            "run_id": self.run_id or "",
            "segment_index": self.segment_index,
            "is_simulated": str(self.is_simulated).lower(),
            "command_id": self.command_id,
            "target_units": self.target_units,
            "start_battery_v": self.start_battery_v,
            "prev_cmd_id": self.prev_cmd_id,
            "prev_residual_angle": self.prev_residual_angle,
            "calibrations_count": self.calibrations_count,
            "total_calib_time": self.total_calib_time,
            "avg_drift_angle": self.avg_drift_angle,
            "actual_time_consumed": self.actual_time_consumed,
            "actual_dist_reached": self.actual_dist_reached,
            "error_cm_or_deg": self.error_cm_or_deg,
            "active_champion_id": self.active_champion_id,
            "shadow_predictions_json": self.shadow_predictions_json,
        }

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "RunRecord":
        run_id = int(row["run_id"]) if row.get("run_id") else None
        return cls(
            run_id=run_id,
            segment_index=int(row["segment_index"]),
            is_simulated=row["is_simulated"].strip().lower() == "true",
            command_id=int(row["command_id"]),
            target_units=float(row["target_units"]),
            start_battery_v=float(row["start_battery_v"]),
            prev_cmd_id=int(row["prev_cmd_id"]),
            prev_residual_angle=float(row["prev_residual_angle"]),
            calibrations_count=int(row["calibrations_count"]),
            total_calib_time=float(row["total_calib_time"]),
            avg_drift_angle=float(row["avg_drift_angle"]),
            actual_time_consumed=float(row["actual_time_consumed"]),
            actual_dist_reached=float(row["actual_dist_reached"]),
            error_cm_or_deg=float(row["error_cm_or_deg"]),
            active_champion_id=row.get("active_champion_id", ""),
            shadow_predictions_json=row.get("shadow_predictions_json", "{}"),
        )


class MockPhysicsEnv:
    """Digital twin used for simulation and production rehearsal data."""

    def __init__(self, env: EnvHelper):
        seed = env.get_val("RANDOM_SEED", int, default=42)
        self._random = random.Random(seed)
        self.speed = env.get_val("PHYSICS_SPEED", float, required=True)
        self.turn_speed_deg_per_sec = env.get_val("TURN_SPEED_DEG_PER_SEC", float, default=90.0)
        self.battery_decay_exponent = env.get_val("BATTERY_DECAY_EXPONENT", float, required=True)
        self.angular_drift_factor = env.get_val("ANGULAR_DRIFT_FACTOR", float, required=True)
        self.payload_factor = env.get_val("PAYLOAD_FACTOR", float, required=True)
        self.noise_std = env.get_val("NOISE_STD", float, required=True)
        self.real_world_bias = env.get_val("REAL_WORLD_BIAS", float, required=True)
        self.distance_min = env.get_val("DISTANCE_MIN", float, required=True)
        self.distance_max = env.get_val("DISTANCE_MAX", float, required=True)
        self.battery_min = env.get_val("BATTERY_MIN", float, required=True)
        self.battery_max = env.get_val("BATTERY_MAX", float, required=True)
        self.heading_min = env.get_val("HEADING_MIN", float, required=True)
        self.heading_max = env.get_val("HEADING_MAX", float, required=True)
        self.terrain_min = env.get_val("TERRAIN_MIN", float, required=True)
        self.terrain_max = env.get_val("TERRAIN_MAX", float, required=True)
        self.payload_min = env.get_val("PAYLOAD_MIN", float, required=True)
        self.payload_max = env.get_val("PAYLOAD_MAX", float, required=True)
        self.route_segment_count = env.get_val("ROUTE_SEGMENT_COUNT", int, default=7)
        if self.route_segment_count <= 0:
            raise ValueError("ROUTE_SEGMENT_COUNT must be at least 1")
        self.turn_target_min_deg = env.get_val("TURN_TARGET_MIN_DEG", float, default=45.0)
        self.turn_target_max_deg = env.get_val("TURN_TARGET_MAX_DEG", float, default=90.0)

    def generate_runs(self, count: int, is_simulated: bool) -> list[RunRecord]:
        runs: list[RunRecord] = []
        prev_cmd_id = 0
        prev_residual_angle = 0.0
        for index in range(count):
            segment_index = (index % self.route_segment_count) + 1
            run, prev_cmd_id, prev_residual_angle = self.generate_run(
                is_simulated=is_simulated,
                segment_index=segment_index,
                prev_cmd_id=prev_cmd_id,
                prev_residual_angle=prev_residual_angle,
            )
            runs.append(run)
        return runs

    def generate_run(
        self,
        is_simulated: bool,
        segment_index: int = 1,
        prev_cmd_id: int = 0,
        prev_residual_angle: float = 0.0,
    ) -> tuple[RunRecord, int, float]:
        command_id = self._random.choice([COMMAND_FORWARD, COMMAND_TURN_LEFT, COMMAND_TURN_RIGHT])
        target_units = (
            self._random.uniform(self.distance_min, self.distance_max)
            if command_id == COMMAND_FORWARD
            else self._random.uniform(self.turn_target_min_deg, self.turn_target_max_deg)
        )
        start_battery_v = self._random.uniform(self.battery_min, self.battery_max)
        calibrations_count = self._random.randint(0, 2 if is_simulated else 4)
        drift_base = self._random.uniform(self.heading_min, self.heading_max)
        avg_drift_angle = drift_base + (prev_residual_angle * PREV_RESIDUAL_DRIFT_FACTOR)
        total_calib_time = calibrations_count * self._random.uniform(0.03, 0.25)
        terrain_factor = self._random.uniform(self.terrain_min, self.terrain_max)
        payload = self._random.uniform(self.payload_min, self.payload_max)
        expected_time = self._expected_time(command_id, target_units)
        battery_penalty = expected_time * ((1 / max(start_battery_v, 0.05) ** self.battery_decay_exponent) - 1)
        angular_penalty = (
            abs(avg_drift_angle + prev_residual_angle * ADDITIONAL_RESIDUAL_ANGULAR_FACTOR) * self.angular_drift_factor
        )
        payload_penalty = payload * self.payload_factor
        environment_bias = 0.0 if is_simulated else self.real_world_bias
        noise = self._random.gauss(0.0, self.noise_std * (1.0 if is_simulated else 1.5))
        actual_time_consumed = max(
            MIN_EXPECTED_TIME_SECONDS,
            expected_time
            + battery_penalty
            + angular_penalty
            + payload_penalty
            + terrain_factor
            + total_calib_time
            + environment_bias
            + noise,
        )
        command_error_scale = FORWARD_ERROR_SCALE if command_id == COMMAND_FORWARD else TURN_ERROR_SCALE
        noise_scale = 0.6 if is_simulated else 1.0
        error_cm_or_deg = self._random.gauss(
            prev_residual_angle * PREV_RESIDUAL_GAUSSIAN_MEAN_FACTOR,
            max(command_error_scale * target_units * noise_scale, 0.001),
        )
        actual_dist_reached = target_units - error_cm_or_deg
        next_residual_angle = (
            error_cm_or_deg
            if command_id in TURN_COMMAND_IDS
            else (prev_residual_angle * PREV_RESIDUAL_DECAY) + (avg_drift_angle * DRIFT_CONTRIBUTION)
        )
        return RunRecord(
            segment_index=segment_index,
            is_simulated=is_simulated,
            command_id=command_id,
            target_units=target_units,
            start_battery_v=start_battery_v,
            prev_cmd_id=prev_cmd_id,
            prev_residual_angle=prev_residual_angle,
            calibrations_count=calibrations_count,
            total_calib_time=total_calib_time,
            avg_drift_angle=avg_drift_angle,
            actual_time_consumed=actual_time_consumed,
            actual_dist_reached=actual_dist_reached,
            error_cm_or_deg=error_cm_or_deg,
            forward_units_per_second=self.speed,
            turn_degrees_per_second=self.turn_speed_deg_per_sec,
        ), command_id, next_residual_angle

    def _expected_time(self, command_id: int, target_units: float) -> float:
        if command_id == COMMAND_FORWARD:
            return max(MIN_EXPECTED_TIME_SECONDS, target_units / max(self.speed, MIN_FORWARD_SPEED_DIVISOR))
        return max(MIN_EXPECTED_TIME_SECONDS, target_units / max(self.turn_speed_deg_per_sec, MIN_TURN_SPEED_DIVISOR))


def write_run_history(path: Path, records: Iterable[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row())


def read_run_history(path: Path) -> list[RunRecord]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [RunRecord.from_csv_row(row) for row in reader]
