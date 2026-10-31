import pytest

from tests.oracle import Oracle
from warehouse_planning.instances import load_split
from warehouse_planning.planner import load_optimal

BENCHMARK = load_split("benchmark")
OPTIMAL = load_optimal("benchmark")
# Exhaustive search stays fast up to here; the state space explodes with a second robot and more items.
EXHAUSTIVE = [i for i in BENCHMARK if i.tier <= 4]


@pytest.mark.parametrize("instance", EXHAUSTIVE, ids=lambda i: i.name)
def test_planner_lengths_are_truly_optimal(instance):
    assert OPTIMAL[instance.name]["length"] == Oracle(instance).shortest_plan_length()


def test_every_problem_has_a_reference_plan():
    assert {i.name for i in BENCHMARK} == set(OPTIMAL)
