import re

from warehouse_planning.instances import SPLIT_SEEDS, TIERS, generate_split, load_split, worked_examples


def test_generation_is_deterministic():
    first, second = generate_split("benchmark"), generate_split("benchmark")
    assert [i.pddl for i in first] == [i.pddl for i in second]


def test_saved_problems_match_the_generator():
    examples = {i.fingerprint for i in worked_examples()}
    saved = {i.name: i.pddl for i in load_split("benchmark")}
    assert saved == {i.name: i.pddl for i in generate_split("benchmark", exclude=examples)}


def test_splits_never_share_a_task():
    names = [*SPLIT_SEEDS, "examples"]
    fingerprints = {split: {i.fingerprint for i in load_split(split)} for split in names}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            assert not fingerprints[first] & fingerprints[second], (first, second)


def test_unseen_maps_appear_nowhere_else():
    def maps(split):
        return {frozenset(i.coords) for i in load_split(split)}

    unseen = maps("unseen")
    assert unseen
    for split in ("benchmark", "dev", "train", "examples"):
        assert not unseen & maps(split), split


def test_benchmark_covers_every_tier_evenly():
    tiers = [i.tier for i in load_split("benchmark")]
    assert len(tiers) == 30
    assert all(tiers.count(t.level) == 5 for t in TIERS)


def test_map_is_symmetric_and_shelves_are_side_entered():
    for instance in load_split("benchmark"):
        edges = set(re.findall(r"\(connected ([^\s()]+) ([^\s()]+)\)", instance.pddl))
        assert all((b, a) in edges for a, b in edges)
        for a, b in edges:
            if a.startswith("shelf"):
                assert instance.coords[a][1] == instance.coords[b][1]


def test_only_robot_locations_start_occupied():
    for instance in load_split("benchmark"):
        clear = set(re.findall(r"\(clear ([^\s()]+)\)", instance.pddl))
        assert clear == set(instance.coords) - set(instance.robots.values())
