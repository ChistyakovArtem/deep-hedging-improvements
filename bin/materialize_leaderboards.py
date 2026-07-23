from __future__ import annotations

import argparse
import json

from src.config import PROJECT_ROOT, load_project_config
from src.market import materialize_leaderboard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--board", choices=["public", "private", "all"], default="all"
    )
    args = parser.parse_args()
    project = load_project_config()
    names = ("public", "private") if args.board == "all" else (args.board,)
    result = {}
    for name in names:
        _, metadata = materialize_leaderboard(
            PROJECT_ROOT,
            project["market"],
            project["leaderboards"][name],
        )
        result[name] = metadata
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

