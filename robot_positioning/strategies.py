from __future__ import annotations

from abc import ABC, abstractmethod

from .simulation import RunRecord


class EstimationStrategy(ABC):
    short_name = "Base"

    @abstractmethod
    def training_target(self, record: RunRecord) -> float:
        raise NotImplementedError

    @abstractmethod
    def prediction_to_time(self, record: RunRecord, model_output: float) -> float:
        raise NotImplementedError


class DirectStrategy(EstimationStrategy):
    short_name = "Direct"

    def training_target(self, record: RunRecord) -> float:
        return record.actual_time_total

    def prediction_to_time(self, record: RunRecord, model_output: float) -> float:
        return max(0.0, model_output)


class ResidualStrategy(EstimationStrategy):
    short_name = "Residual"

    def __init__(self, tile_time: float, turn_time: float):
        self.tile_time = tile_time
        self.turn_time = turn_time

    def baseline_time(self, record: RunRecord) -> float:
        return (record.total_tiles * self.tile_time) + (record.num_corners * self.turn_time)

    def training_target(self, record: RunRecord) -> float:
        return record.actual_time_total - self.baseline_time(record)

    def prediction_to_time(self, record: RunRecord, model_output: float) -> float:
        return max(0.0, self.baseline_time(record) + model_output)
