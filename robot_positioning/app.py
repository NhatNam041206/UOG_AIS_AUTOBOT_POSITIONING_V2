from __future__ import annotations

import argparse

from .config import EnvHelper
from .controller import TournamentManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot positioning estimation engine")
    parser.add_argument("--env-file", default=".env", help="Path to the .env file")
    parser.add_argument("--production-runs", type=int, default=None, help="Override number of production runs")
    args = parser.parse_args()

    env = EnvHelper(args.env_file)
    manager = TournamentManager(env)
    mode = env.get_val("APP_MODE", str, default="EXPERIMENT").upper()
    if mode == "EXPERIMENT":
        champion = manager.run_experiment()
        print(f"Experimental champion: {champion}")
        return
    if args.production_runs is not None:
        manager.run_production(manager.physics_env.generate_runs(args.production_runs, is_simulated=False))
        return
    manager.run_production()


if __name__ == "__main__":
    main()
