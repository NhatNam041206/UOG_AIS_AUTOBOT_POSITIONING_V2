from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import EnvHelper

CSV_FIELDNAMES = [
    "run_id",
    "total_tiles",
    "num_corners",
    "start_battery_v",
    "calibrations_total",
    "actual_time_total",
    "endpoint_error_cm",
    "endpoint_deviated_deg",
    "is_simulated",
]
MIN_ACTUAL_TIME = 0.001
MIN_BATTERY_VOLTAGE = 0.01


@dataclass
class RunRecord:
    run_id: int | None = None
    total_tiles: int = 20
    num_corners: int = 4
    start_battery_v: float = 12.0
    calibrations_total: int = 1
    actual_time_total: float = 0.0
    endpoint_error_cm: float = 0.0
    endpoint_deviated_deg: float = 0.0
    is_simulated: bool = True

    def __post_init__(self) -> None:
        if self.total_tiles <= 0:
            raise ValueError("total_tiles must be positive")
        if self.num_corners <= 0:
            raise ValueError("num_corners must be positive")
        if self.start_battery_v <= 0:
            raise ValueError("start_battery_v must be positive")
        if self.calibrations_total <= 0:
            raise ValueError("calibrations_total must be positive")
        if self.actual_time_total < 0:
            raise ValueError("actual_time_total cannot be negative")

    def features(self) -> list[float]:
        return [
            float(self.total_tiles),
            float(self.num_corners),
            self.start_battery_v,
            float(self.calibrations_total),
        ]

    def to_csv_row(self) -> dict[str, str | float | int]:
        return {
            "run_id": self.run_id or "",
            "total_tiles": self.total_tiles,
            "num_corners": self.num_corners,
            "start_battery_v": self.start_battery_v,
            "calibrations_total": self.calibrations_total,
            "actual_time_total": self.actual_time_total,
            "endpoint_error_cm": self.endpoint_error_cm,
            "endpoint_deviated_deg": self.endpoint_deviated_deg,
            "is_simulated": str(self.is_simulated).lower(),
        }

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "RunRecord":
        run_id = int(row["run_id"]) if row.get("run_id") else None
        return cls(
            run_id=run_id,
            total_tiles=int(row["total_tiles"]),
            num_corners=int(row["num_corners"]),
            start_battery_v=float(row["start_battery_v"]),
            calibrations_total=int(row["calibrations_total"]),
            actual_time_total=float(row["actual_time_total"]),
            endpoint_error_cm=float(row["endpoint_error_cm"]),
            endpoint_deviated_deg=float(row["endpoint_deviated_deg"]),
            is_simulated=row["is_simulated"].strip().lower() == "true",
        )


class MockPhysicsEnv:
    """Digital twin for route-time and endpoint behavior."""

    def __init__(self, env: EnvHelper):
        seed = env.get_val("RANDOM_SEED", int, default=42)
        self._random = random.Random(seed)
        self.tile_time = env.get_val("PHYSICS_TILE_TIME", float, required=True)
        self.turn_time = env.get_val("PHYSICS_TURN_TIME", float, required=True)
        self.nominal_voltage = env.get_val("PHYSICS_NOMINAL_VOLTAGE", float, required=True)
        self.battery_exponent = env.get_val("BATTERY_DECAY_EXPONENT", float, required=True)
        self.noise_std_seconds = env.get_val("TIME_NOISE_STD", float, required=True)
        self.noise_std_seconds_real = env.get_val("TIME_NOISE_STD_REAL", float, required=True)

        self.tiles_min = env.get_val("TOTAL_TILES_MIN", int, required=True)
        self.tiles_max = env.get_val("TOTAL_TILES_MAX", int, required=True)
        self.corners_min = env.get_val("NUM_CORNERS_MIN", int, required=True)
        self.corners_max = env.get_val("NUM_CORNERS_MAX", int, required=True)
        self.battery_min = env.get_val("START_BATTERY_MIN", float, required=True)
        self.battery_max = env.get_val("START_BATTERY_MAX", float, required=True)
        self.calibration_min = env.get_val("CALIBRATIONS_MIN", int, required=True)
        self.calibration_max = env.get_val("CALIBRATIONS_MAX", int, required=True)

        self.endpoint_error_abs_max = env.get_val("ENDPOINT_ERROR_ABS_MAX_CM", float, required=True)
        self.endpoint_deviated_abs_max = env.get_val("ENDPOINT_DEVIATION_ABS_MAX_DEG", float, required=True)
        self.real_world_error_multiplier = env.get_val("REAL_WORLD_ERROR_MULTIPLIER", float, required=True)

    def baseline_time(self, total_tiles: int, num_corners: int) -> float:
        return (total_tiles * self.tile_time) + (num_corners * self.turn_time)

    def generate_runs(self, count: int, is_simulated: bool) -> list[RunRecord]:
        return [self.generate_run(is_simulated=is_simulated) for _ in range(count)]

    def generate_run(self, is_simulated: bool) -> RunRecord:
        total_tiles = self._random.randint(self.tiles_min, self.tiles_max)
        num_corners = self._random.randint(self.corners_min, self.corners_max)
        start_battery_v = self._random.uniform(self.battery_min, self.battery_max)
        calibrations_total = self._random.randint(self.calibration_min, self.calibration_max)

        baseline = self.baseline_time(total_tiles=total_tiles, num_corners=num_corners)
        voltage_factor = (self.nominal_voltage / max(start_battery_v, MIN_BATTERY_VOLTAGE)) ** self.battery_exponent
        noise_std = self.noise_std_seconds if is_simulated else self.noise_std_seconds_real
        noise = self._random.gauss(0.0, noise_std)
        actual_time_total = max(MIN_ACTUAL_TIME, (baseline * voltage_factor) + noise)

        calibration_damp = 1.0 / max(calibrations_total, 1)
        error_scale = self.endpoint_error_abs_max * calibration_damp
        angle_scale = self.endpoint_deviated_abs_max * calibration_damp
        if not is_simulated:
            error_scale *= self.real_world_error_multiplier
            angle_scale *= self.real_world_error_multiplier

        endpoint_error_cm = self._random.uniform(-error_scale, error_scale)
        endpoint_deviated_deg = self._random.uniform(-angle_scale, angle_scale)

        return RunRecord(
            total_tiles=total_tiles,
            num_corners=num_corners,
            start_battery_v=start_battery_v,
            calibrations_total=calibrations_total,
            actual_time_total=actual_time_total,
            endpoint_error_cm=endpoint_error_cm,
            endpoint_deviated_deg=endpoint_deviated_deg,
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
