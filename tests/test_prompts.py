import re

from warehouse_planning.instances import load_split
from warehouse_planning.prompts import describe, planning_messages, repair_messages

INSTANCE = load_split("benchmark")[6]


def test_description_carries_everything_the_problem_says():
    text = describe(INSTANCE.pddl)
    for robot, location in INSTANCE.robots.items():
        assert f"{robot} is at {location}" in text
    for item, shelf in INSTANCE.items.items():
        assert f"{item} is on {shelf}" in text
    for item, station in INSTANCE.goals.items():
        assert f"{item} to {station}" in text


def test_map_lists_every_connection_once():
    lines = describe(INSTANCE.pddl).split("Map (location: where a robot can move from it):\n")[1].splitlines()
    listed = {(line.split(": ")[0], target) for line in lines for target in line.split(": ")[1].split()}
    edges = set(re.findall(r"\(connected ([^\s()]+) ([^\s()]+)\)", INSTANCE.pddl))
    assert listed == edges


def test_examples_are_valid_plans_for_their_own_problems():
    from warehouse_planning import DOMAIN_PATH
    from warehouse_planning.planner import load_optimal
    from warehouse_planning.validator import validate

    for instance in load_split("examples"):
        plan = "\n".join(load_optimal("examples")[instance.name]["plan"])
        assert validate(DOMAIN_PATH, instance.pddl, plan).valid


def test_benchmark_problem_is_never_a_worked_example():
    messages = planning_messages(INSTANCE.pddl)
    assert messages[-1]["content"].startswith(describe(INSTANCE.pddl))
    assert not any(describe(INSTANCE.pddl) in m["content"] for m in messages[:-1])


STATE_MID_PLAN = ["(at robot1 a21)", "(holding robot1 item1)", "(stored item2 shelf1)", "(delivered item3 station1)"]


def test_state_renders_as_a_smaller_problem():
    text = describe(INSTANCE.pddl, STATE_MID_PLAN)
    assert "robot1 is at a21, holding item1" in text
    assert "item2 is on shelf1" in text
    assert "item1 is on" not in text


def test_reached_goals_leave_the_description():
    goal = next(iter(INSTANCE.goals.items()))
    state = [f"(at robot1 a21)", f"(delivered {goal[0]} {goal[1]})"]
    assert f"{goal[0]} to {goal[1]}" not in describe(INSTANCE.pddl, state)


def test_repair_is_a_new_problem_from_the_reached_state():
    messages = repair_messages(INSTANCE.pddl, STATE_MID_PLAN, "Step 2: nope.")
    request = messages[-1]["content"]
    assert request.startswith(describe(INSTANCE.pddl, STATE_MID_PLAN))
    assert "Step 2: nope." in request
    assert request.endswith("Plan:")
    assert describe(INSTANCE.pddl) not in request


def test_repair_shows_how_to_continue_mid_plan():
    messages = repair_messages(INSTANCE.pddl, STATE_MID_PLAN, "x")
    assert any("holding item1" in m["content"] for m in messages[:-1] if m["role"] == "user")
    first_request = planning_messages(INSTANCE.pddl)
    assert len(messages) == len(first_request) + 2
