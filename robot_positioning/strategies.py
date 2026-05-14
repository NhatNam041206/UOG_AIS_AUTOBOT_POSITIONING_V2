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
        return record.actual_time

    def prediction_to_time(self, record: RunRecord, model_output: float) -> float:
        return max(0.0, model_output)


class ResidualStrategy(EstimationStrategy):
    short_name = "Residual"

    def calculate_residual(self, record: RunRecord) -> float:
        return record.actual_time - record.baseline_time

    def training_target(self, record: RunRecord) -> float:
        return self.calculate_residual(record)

    def prediction_to_time(self, record: RunRecord, model_output: float) -> float:
        return max(0.0, record.baseline_time + model_output)
