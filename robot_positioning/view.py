from __future__ import annotations


class TerminalDashboard:
    """Simple terminal-oriented dashboard output."""

    def render(
        self,
        champion: str,
        start_battery_v: float,
        active_model: str,
        predictions: dict[str, float],
        time_distribution: dict[str, float] | None = None,
    ) -> str:
        shadow_predictions = {name: value for name, value in sorted(predictions.items()) if name != active_model}
        lines = [
            "=== Robot Endpoint Estimation Dashboard ===",
            f"Champion: {champion}",
            f"Active Model: {active_model}",
            f"Start Battery: {start_battery_v:.2f}V",
            "Shadow Predictions:",
        ]
        if shadow_predictions:
            lines.extend(f"- {name}: {value:.3f}s" for name, value in shadow_predictions.items())
        else:
            lines.append("- None")

        if time_distribution:
            lines.extend(
                [
                    "Time Distribution:",
                    f"- Forward Total: {time_distribution['forward_time_total']:.3f}s",
                    f"- Turning Total: {time_distribution['turn_time_total']:.3f}s",
                    f"- Forward/Tile: {time_distribution['forward_time_per_tile']:.3f}s",
                    f"- Turn/Corner: {time_distribution['turn_time_per_corner']:.3f}s",
                ]
            )

        output = "\n".join(lines)
        print(output)
        return output
