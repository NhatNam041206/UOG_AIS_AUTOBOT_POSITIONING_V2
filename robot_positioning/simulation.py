from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import EnvHelper

FEATURE_NAMES = ["distance", "battery_level", "heading_error", "terrain_factor", "payload"]
CSV_FIELDNAMES = ["run_id", *FEATURE_NAMES, "baseline_time", "actual_time", "is_simulated", "active_model"]


@dataclass
class RunRecord:
    distance: float
    battery_level: float
    heading_error: float
    terrain_factor: float
    payload: float
    baseline_time: float
    actual_time: float
    is_simulated: bool
    run_id: int | None = None
    active_model: str = ""

    def features(self) -> list[float]:
        return [self.distance, self.battery_level, self.heading_error, self.terrain_factor, self.payload]

    def to_csv_row(self) -> dict[str, str | float | int]:
        return {
            "run_id": self.run_id or "",
            "distance": self.distance,
            "battery_level": self.battery_level,
            "heading_error": self.heading_error,
            "terrain_factor": self.terrain_factor,
            "payload": self.payload,
            "baseline_time": self.baseline_time,
            "actual_time": self.actual_time,
            "is_simulated": str(self.is_simulated).lower(),
            "active_model": self.active_model,
        }

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "RunRecord":
        run_id = int(row["run_id"]) if row.get("run_id") else None
        return cls(
            run_id=run_id,
            distance=float(row["distance"]),
            battery_level=float(row["battery_level"]),
            heading_error=float(row["heading_error"]),
            terrain_factor=float(row["terrain_factor"]),
            payload=float(row["payload"]),
            baseline_time=float(row["baseline_time"]),
            actual_time=float(row["actual_time"]),
            is_simulated=row["is_simulated"].strip().lower() == "true",
            active_model=row.get("active_model", ""),
        )


class MockPhysicsEnv:
    """Digital twin used for simulation and production rehearsal data."""

    def __init__(self, env: EnvHelper):
        seed = env.get_val("RANDOM_SEED", int, default=42)
        self._random = random.Random(seed)
        self.speed = env.get_val("PHYSICS_SPEED", float, required=True)
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

    def generate_runs(self, count: int, is_simulated: bool) -> list[RunRecord]:
        return [self.generate_run(is_simulated=is_simulated) for _ in range(count)]

    def generate_run(self, is_simulated: bool) -> RunRecord:
        distance = self._random.uniform(self.distance_min, self.distance_max)
        battery_level = self._random.uniform(self.battery_min, self.battery_max)
        heading_error = self._random.uniform(self.heading_min, self.heading_max)
        terrain_factor = self._random.uniform(self.terrain_min, self.terrain_max)
        payload = self._random.uniform(self.payload_min, self.payload_max)
        baseline_time = distance / self.speed
        battery_penalty = baseline_time * ((1 / max(battery_level, 0.05) ** self.battery_decay_exponent) - 1)
        angular_penalty = abs(heading_error) * self.angular_drift_factor
        payload_penalty = payload * self.payload_factor
        environment_bias = 0.0 if is_simulated else self.real_world_bias
        noise = self._random.gauss(0.0, self.noise_std * (1.0 if is_simulated else 1.5))
        actual_time = max(
            0.1,
            baseline_time + battery_penalty + angular_penalty + payload_penalty + terrain_factor + environment_bias + noise,
        )
        return RunRecord(
            distance=distance,
            battery_level=battery_level,
            heading_error=heading_error,
            terrain_factor=terrain_factor,
            payload=payload,
            baseline_time=baseline_time,
            actual_time=actual_time,
            is_simulated=is_simulated,
        )


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
