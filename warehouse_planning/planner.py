"""Reference plans from Fast Downward, cached in pddl/<split>/optimal.json."""

import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import unified_planning.shortcuts as up

from warehouse_planning import DOMAIN_PATH, PDDL_DIR
from warehouse_planning.instances import Instance, load_split
from warehouse_planning.validator import format_action, parse_problem

up.get_environment().credits_stream = None

SEARCH_TIMEOUT_SECONDS = 1800
WORKERS = 4


def solve_optimal(instance: Instance, timeout: float = SEARCH_TIMEOUT_SECONDS) -> dict:
    """Solve with A* + LM-cut. Moves have unit cost, so `length` is the minimum number of actions.

    up-fast-downward reports SOLVED_SATISFICING for problems without a metric even though
    astar(lmcut()) is optimal; tests/test_planner.py checks the lengths against an exhaustive search.
    """
    problem = parse_problem(DOMAIN_PATH, instance.pddl)
    start = time.perf_counter()
    with up.OneshotPlanner(name="fast-downward-opt") as planner:
        result = planner.solve(problem, timeout=timeout)
    seconds = time.perf_counter() - start
    if result.plan is None:
        raise RuntimeError(f"{instance.name}: Fast Downward found no plan ({result.status.name})")
    plan = [format_action(a) for a in result.plan.actions]
    return {"length": len(plan), "plan": plan, "seconds": round(seconds, 2)}


def _isolate_working_directory() -> None:
    """Fast Downward writes output.sas to the working directory, so parallel runs must not share one."""
    os.chdir(tempfile.mkdtemp(prefix="fast-downward-"))


def solve_split(split: str) -> dict[str, dict]:
    """Solve every unsolved problem of a split, saving each result as it finishes."""
    path = PDDL_DIR / split / "optimal.json"
    solved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    pending = [i for i in load_split(split) if i.name not in solved]
    with ProcessPoolExecutor(WORKERS, initializer=_isolate_working_directory) as pool:
        futures = {pool.submit(solve_optimal, instance): instance for instance in pending}
        for future in as_completed(futures):
            instance = futures[future]
            solved[instance.name] = entry = future.result()
            print(f"{split}/{instance.name} tier {instance.tier}: {entry['length']} steps in {entry['seconds']}s", flush=True)
            path.write_text(json.dumps(dict(sorted(solved.items())), indent=1), encoding="utf-8")
    return solved


def load_optimal(split: str) -> dict[str, dict]:
    return json.loads((PDDL_DIR / split / "optimal.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    import sys

    for split_name in sys.argv[1:] or ["benchmark", "dev", "train"]:
        solve_split(split_name)
