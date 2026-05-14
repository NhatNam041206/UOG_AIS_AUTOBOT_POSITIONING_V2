from __future__ import annotations


class TerminalDashboard:
    """Simple terminal-oriented dashboard output."""

    def render(self, champion: str, battery_level: float, active_model: str, predictions: dict[str, float]) -> str:
        shadow_predictions = {
            name: value for name, value in sorted(predictions.items()) if name != champion
        }
        lines = [
            "=== Robot Positioning Dashboard ===",
            f"Champion: {champion}",
            f"Active Model: {active_model}",
            f"Start Battery: {battery_level:.2f}V",
            "Shadow Predictions:",
        ]
        if shadow_predictions:
            lines.extend(f"- {name}: {value:.3f}s" for name, value in shadow_predictions.items())
        else:
            lines.append("- None")
        output = "\n".join(lines)
        print(output)
        return output
