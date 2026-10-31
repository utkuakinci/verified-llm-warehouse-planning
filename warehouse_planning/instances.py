"""Generate warehouse problems: a benchmark from easy to hard, plus disjoint dev and train sets.

Layout, for 2 shelf columns and 2 shelf rows (`.` aisle, `S` shelf, `T` station):

    . . . . .
    . S . S .
    . S . S .
    . . . . .
      T   T

Shelves connect to the cells on their left and right, and stations are dead ends below the bottom
aisle. Robots enter them to pick or drop, so a robot parked on a shelf or in an aisle can block
another one.
"""

import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from warehouse_planning import PDDL_DIR


@dataclass(frozen=True)
class Tier:
    level: int
    robots: int
    items: int
    stations: int
    shelf_columns: int
    shelf_rows: int


# Ordered by mean optimal plan length. Three robots with four items is out of reach: A* + LM-cut does
# not finish within minutes, and every problem here needs a proven optimal length as its reference.
TIERS = (
    Tier(1, robots=1, items=1, stations=2, shelf_columns=2, shelf_rows=1),
    Tier(2, robots=1, items=2, stations=2, shelf_columns=2, shelf_rows=1),
    Tier(3, robots=2, items=2, stations=1, shelf_columns=2, shelf_rows=2),
    Tier(4, robots=2, items=3, stations=2, shelf_columns=2, shelf_rows=2),
    Tier(5, robots=3, items=3, stations=2, shelf_columns=2, shelf_rows=2),
    Tier(6, robots=2, items=4, stations=2, shelf_columns=2, shelf_rows=2),
)
INSTANCES_PER_TIER = 5

# The same task sizes on wider maps. The prompts and the training data never show these maps, so
# this split tells a model that learned to plan apart from one that memorized the known maps.
UNSEEN_MAP_TIERS = tuple(replace(tier, shelf_columns=3) for tier in TIERS[:4])

SPLIT_SEEDS = {"benchmark": 0, "dev": 100_000, "train": 200_000, "unseen": 300_000}


@dataclass
class Instance:
    name: str
    tier: int
    pddl: str
    # Location name -> [x, y], and everything else the animation needs to draw the warehouse.
    coords: dict[str, list[int]]
    robots: dict[str, str]
    items: dict[str, str]
    goals: dict[str, str]

    @property
    def fingerprint(self) -> str:
        """Identifies the task itself, independent of the instance name."""
        task = {"coords": self.coords, "robots": self.robots, "items": self.items, "goals": self.goals}
        return hashlib.sha1(json.dumps(task, sort_keys=True).encode()).hexdigest()


def _build_graph(tier: Tier):
    """Return (coords, undirected edges, aisle names, shelf names, station names)."""
    width = 2 * tier.shelf_columns + 1
    bottom = tier.shelf_rows + 1
    names: dict[tuple[int, int], str] = {}
    aisles, shelves, stations = [], [], []

    for y in range(bottom + 1):
        for x in range(width):
            if 0 < y < bottom and x % 2 == 1:
                name = f"shelf{len(shelves) + 1}"
                shelves.append(name)
            else:
                name = f"a{x}{y}"
                aisles.append(name)
            names[x, y] = name

    station_columns = _spread(tier.stations, range(1, width, 2))
    for x in station_columns:
        name = f"station{len(stations) + 1}"
        stations.append(name)
        names[x, bottom + 1] = name

    shelf_set = set(shelves)
    edges = []
    for (x, y), name in names.items():
        right, below = names.get((x + 1, y)), names.get((x, y + 1))
        if right:
            edges.append((name, right))
        # Shelves are entered sideways from the aisles only.
        if below and name not in shelf_set and below not in shelf_set:
            edges.append((name, below))

    coords = {name: (x, y) for (x, y), name in names.items()}
    return coords, edges, aisles, shelves, stations


def _spread(count: int, options: range) -> list[int]:
    """Pick `count` evenly spaced values from `options`."""
    options = list(options)
    return [options[i * len(options) // count] for i in range(count)]


def generate(name: str, tier: Tier, rng: random.Random) -> Instance:
    coords, edges, aisles, shelves, stations = _build_graph(tier)
    robot_names = [f"robot{i + 1}" for i in range(tier.robots)]
    item_names = [f"item{i + 1}" for i in range(tier.items)]

    robots = dict(zip(robot_names, rng.sample(aisles, tier.robots)))
    items = dict(zip(item_names, rng.sample(shelves, tier.items)))
    goals = {item: rng.choice(stations) for item in item_names}

    init = [f"(at {r} {loc})" for r, loc in robots.items()]
    init += [f"(empty {r})" for r in robot_names]
    init += [f"(stored {item} {shelf})" for item, shelf in items.items()]
    occupied = set(robots.values())
    init += [f"(clear {loc})" for loc in coords if loc not in occupied]
    for a, b in edges:
        init += [f"(connected {a} {b})", f"(connected {b} {a})"]

    goal = [f"(delivered {item} {station})" for item, station in goals.items()]

    newline = "\n    "
    pddl = f"""(define (problem {name})
  (:domain warehouse)
  (:objects
    {' '.join(robot_names)} - robot
    {' '.join(item_names)} - item
    {' '.join(aisles)} - aisle
    {' '.join(shelves)} - shelf
    {' '.join(stations)} - station)
  (:init
    {newline.join(init)})
  (:goal (and
    {newline.join(goal)})))
"""
    return Instance(
        name=name,
        tier=tier.level,
        pddl=pddl,
        coords={loc: list(xy) for loc, xy in coords.items()},
        robots=robots,
        items=items,
        goals=goals,
    )


def generate_split(
    split: str,
    per_tier: int = INSTANCES_PER_TIER,
    exclude: set[str] = frozenset(),
    tiers: tuple[Tier, ...] = TIERS,
) -> list[Instance]:
    """Generate one split, skipping tasks in `exclude` so splits never share a task."""
    rng = random.Random(SPLIT_SEEDS[split])
    instances, seen = [], set(exclude)
    prefix = {"benchmark": "p", "dev": "dev", "train": "train", "unseen": "unseen"}[split]
    for tier in tiers:
        made = 0
        attempts = 0
        while made < per_tier:
            attempts += 1
            if attempts > 10_000:
                raise RuntimeError(f"tier {tier.level} has too few distinct tasks for {per_tier} {split} instances")
            candidate = generate(f"{prefix}{len(instances) + 1:03d}", tier, rng)
            if candidate.fingerprint in seen:
                continue
            seen.add(candidate.fingerprint)
            instances.append(candidate)
            made += 1
    return instances


def write_split(split: str, instances: list[Instance], directory: Path = PDDL_DIR) -> None:
    out = directory / split
    out.mkdir(parents=True, exist_ok=True)
    for instance in instances:
        (out / f"{instance.name}.pddl").write_text(instance.pddl, encoding="utf-8")
        meta = asdict(instance)
        del meta["pddl"]
        (out / f"{instance.name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_split(split: str, directory: Path = PDDL_DIR) -> list[Instance]:
    out = directory / split
    instances = []
    for path in sorted(out.glob("*.pddl")):
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        instances.append(Instance(pddl=path.read_text(encoding="utf-8"), **meta))
    return instances


def worked_examples() -> list[Instance]:
    """The two problems shown to the model as examples, on the same maps as tiers 3 and 1."""
    return [
        generate("example1", TIERS[2], random.Random(910_003)),
        generate("example2", TIERS[0], random.Random(910_001)),
    ]


def main() -> None:
    examples = worked_examples()
    used = {i.fingerprint for i in examples}
    benchmark = generate_split("benchmark", exclude=used)
    used |= {i.fingerprint for i in benchmark}
    dev = generate_split("dev", per_tier=2, exclude=used)
    used |= {i.fingerprint for i in dev}
    train = generate_split("train", per_tier=20, exclude=used)
    unseen = generate_split("unseen", tiers=UNSEEN_MAP_TIERS)
    splits = (("examples", examples), ("benchmark", benchmark), ("dev", dev), ("train", train), ("unseen", unseen))
    for split, instances in splits:
        write_split(split, instances)
        print(f"{split}: {len(instances)} problems")


if __name__ == "__main__":
    main()
