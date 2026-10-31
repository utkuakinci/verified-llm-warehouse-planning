"""Prompts for planning and for continuing a plan after validator feedback.

The model reads a compact rendering of the PDDL problem and writes PDDL actions, which are verified
against the PDDL problem itself. On the dev set the raw PDDL prompt failed at the first step on 11
of 18 attempts, against 2 of 18 for this rendering.
"""

import re
from collections import defaultdict
from functools import lru_cache

from warehouse_planning import DOMAIN_PATH
from warehouse_planning.instances import load_split
from warehouse_planning.planner import load_optimal
from warehouse_planning.validator import validate

NAME = r"([^\s()]+)"

SYSTEM = """You are a precise planner for warehouse robots. Output only a plan: one PDDL action per line.

Rules:
- (move robot from to) needs robot at `from`, `to` listed next to `from` in the map, and no robot at `to`.
- (pick robot item shelf) needs robot at that shelf, the item stored there, and robot empty.
- (drop robot item station) needs robot at that station and holding the item.
- A robot carries one item at a time, and a location holds one robot at a time.

The first move of each robot must start from where that robot is. Every move must follow the map."""


def describe(problem_pddl: str, state: list[str] | None = None) -> str:
    """Render a PDDL problem as a state, a goal and an adjacency list.

    `state` lists the atoms true at some point of a plan; by default the problem's initial state.
    Goals already reached are left out, so a state halfway through reads like a smaller problem.
    """
    facts = " ".join(state) if state is not None else problem_pddl
    held = dict(re.findall(rf"\(holding {NAME} {NAME}\)", facts))
    robots = [
        f"{robot} is at {location}" + (f", holding {held[robot]}" if robot in held else "")
        for robot, location in re.findall(rf"\(at {NAME} {NAME}\)", facts)
    ]
    items = [f"{item} is on {shelf}" for item, shelf in re.findall(rf"\(stored {NAME} {NAME}\)", facts)]
    delivered = set(re.findall(rf"\(delivered {NAME} {NAME}\)", " ".join(state or [])))
    goals = [goal for goal in re.findall(rf"\(delivered {NAME} {NAME}\)", problem_pddl) if goal not in delivered]
    neighbours = defaultdict(list)
    for origin, destination in re.findall(rf"\(connected {NAME} {NAME}\)", problem_pddl):
        neighbours[origin].append(destination)
    return "\n".join(
        [
            f"Robots: {', '.join(robots)}.",
            f"Items: {', '.join(items) or 'none left on the shelves'}.",
            "Goal: " + ", ".join(f"{item} to {station}" for item, station in goals) + ".",
            "Map (location: where a robot can move from it):",
            *(f"{location}: {' '.join(reachable)}" for location, reachable in neighbours.items()),
        ]
    )


@lru_cache(maxsize=None)
def _worked_examples() -> tuple[tuple[str, str], ...]:
    """(problem, plan) pairs. The order matters: the model imitates the last example most closely."""
    plans = load_optimal("examples")
    return tuple((describe(i.pddl), "\n".join(plans[i.name]["plan"])) for i in load_split("examples"))


@lru_cache(maxsize=None)
def _continuation_example() -> tuple[str, str]:
    """The last worked example, picked up right after its robot took the item from the shelf."""
    instance = load_split("examples")[-1]
    plan = load_optimal("examples")[instance.name]["plan"]
    picked = 1 + next(index for index, action in enumerate(plan) if action.startswith("(pick"))
    trace = validate(DOMAIN_PATH, instance.pddl, "\n".join(plan)).trace
    return describe(instance.pddl, trace[picked]), "\n".join(plan[picked:])


def _conversation(examples) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}]
    for problem, plan in examples:
        messages += [
            {"role": "user", "content": f"{problem}\n\nPlan:"},
            {"role": "assistant", "content": plan},
        ]
    return messages


def planning_messages(problem_pddl: str) -> list[dict]:
    """The first request."""
    request = f"{describe(problem_pddl)}\n\nPlan:"
    return [*_conversation(_worked_examples()), {"role": "user", "content": request}]


def repair_messages(problem_pddl: str, state: list[str], feedback: str) -> list[dict]:
    """Restate the task from the state the plan reached, as a new and smaller problem.

    Neither the failed plan nor the initial state appear. Shown its own plan, the model returned it
    almost word for word, and told about a mid-plan state in prose after the initial one, it went
    back to planning from the start.
    """
    request = f"{describe(problem_pddl, state)}\n\nThe last attempt from this state was rejected.\n{feedback}\n\nPlan:"
    examples = (*_worked_examples(), _continuation_example())
    return [*_conversation(examples), {"role": "user", "content": request}]
