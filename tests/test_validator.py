import random

import pytest

from tests.oracle import Oracle
from warehouse_planning import DOMAIN_PATH
from warehouse_planning.instances import load_split
from warehouse_planning.planner import load_optimal
from warehouse_planning.validator import Outcome, extract_actions, validate

BENCHMARK = load_split("benchmark")
OPTIMAL = load_optimal("benchmark")
SIMPLE = BENCHMARK[0]
SIMPLE_PLAN = OPTIMAL[SIMPLE.name]["plan"]


def check(text, instance=SIMPLE):
    return validate(DOMAIN_PATH, instance.pddl, text)


@pytest.mark.parametrize("instance", BENCHMARK, ids=lambda i: i.name)
def test_planner_plans_pass_the_validator(instance):
    result = check("\n".join(OPTIMAL[instance.name]["plan"]), instance)
    assert result.valid, result.feedback
    assert len(result.trace) == len(result.actions) + 1


def test_missing_last_step_reports_the_goal():
    result = check("\n".join(SIMPLE_PLAN[:-1]))
    assert result.outcome is Outcome.GOAL
    assert result.failed_step is None
    assert f"(delivered item1 {SIMPLE.goals['item1']})" in result.feedback


def test_inapplicable_action_names_step_and_precondition():
    plan = SIMPLE_PLAN[:]
    plan.remove(next(a for a in plan if a.startswith("(pick")))
    result = check("\n".join(plan))
    assert result.outcome is Outcome.PRECONDITION
    assert "(holding robot1 item1)" in result.feedback
    assert f"Step {result.failed_step}:" in result.feedback


def test_move_between_unconnected_locations_is_explained():
    robot_at = SIMPLE.robots["robot1"]
    far_away = "station1" if robot_at != "station1" else "station2"
    result = check(f"(move robot1 {robot_at} {far_away})")
    assert result.outcome is Outcome.PRECONDITION
    assert f"(connected {robot_at} {far_away})" in result.feedback
    assert f"From {robot_at}, robot1 can move to:" in result.feedback


@pytest.mark.parametrize(
    "text, expected",
    [
        ("(fly robot1 a00 a10)", "not an action"),
        ("(move robot1)", "has 1 arguments"),
        ("(move robot1 a00 nowhere)", "not an object"),
        ("(pick robot1 item1 station1)", "needs type shelf"),
        ("(move item1 a00 a10)", "needs type robot"),
    ],
)
def test_malformed_actions(text, expected):
    result = check(text)
    assert result.outcome is Outcome.MALFORMED
    assert result.failed_step == 1
    assert expected in result.feedback


@pytest.mark.parametrize("text", ["", "I cannot solve this.", "move the robot to the shelf"])
def test_text_without_actions(text):
    assert check(text).outcome is Outcome.NO_PLAN


def test_extraction_tolerates_formatting():
    text = "Here is the plan:\n```pddl\n1. (move r1 A00 a10)\n; walk on\n(pick r1 i1 shelf1)\n```\n"
    assert extract_actions(text) == [("move", ["r1", "a00", "a10"]), ("pick", ["r1", "i1", "shelf1"])]


def test_extraction_stops_at_prose_after_the_plan():
    text = "(move r1 a00 a10)\nThat should work. Alternatively:\n(move r1 a00 a01)"
    assert extract_actions(text) == [("move", ["r1", "a00", "a10"])]


def _mutate(plan: list[str], rng: random.Random) -> list[str]:
    mutated = plan[:]
    for _ in range(rng.randint(1, 3)):
        kind = rng.choice(["drop", "swap", "duplicate", "truncate"])
        if len(mutated) < 2:
            break
        index = rng.randrange(len(mutated) - 1)
        if kind == "drop":
            del mutated[index]
        elif kind == "swap":
            mutated[index], mutated[index + 1] = mutated[index + 1], mutated[index]
        elif kind == "duplicate":
            mutated.insert(index, mutated[index])
        else:
            mutated = mutated[: index + 1]
    return mutated


@pytest.mark.parametrize("instance", BENCHMARK[::3], ids=lambda i: i.name)
def test_validator_agrees_with_the_independent_rules(instance):
    rng = random.Random(instance.name)
    oracle = Oracle(instance)
    seen = set()
    for _ in range(40):
        plan = _mutate(OPTIMAL[instance.name]["plan"], rng)
        result = check("\n".join(plan), instance)
        expected = oracle.verdict(plan)
        assert result.outcome.value == expected, plan
        seen.add(expected)
    assert "precondition" in seen
