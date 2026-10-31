"""The warehouse rules written a second time, without unified-planning or Fast Downward.

It exists to check those two: the validator's verdicts and the planner's optimality claim both rest
on them, and a wrong verdict would silently corrupt every metric.
"""

import re
from collections import deque
from dataclasses import dataclass

from warehouse_planning.instances import Instance


@dataclass(frozen=True)
class State:
    at: tuple[tuple[str, str], ...]  # (robot, location), sorted
    holding: tuple[tuple[str, str], ...]  # (robot, item), sorted
    stored: tuple[tuple[str, str], ...]  # (item, shelf), sorted
    delivered: tuple[tuple[str, str], ...]  # (item, station), sorted


class Oracle:
    def __init__(self, instance: Instance):
        self.instance = instance
        self.connected = set(re.findall(r"\(connected ([^\s()]+) ([^\s()]+)\)", instance.pddl))
        self.locations = set(instance.coords)
        self.robots = set(instance.robots)
        self.items = set(instance.items)
        self.goal = set(instance.goals.items())

    def initial(self) -> State:
        return State(
            at=tuple(sorted(self.instance.robots.items())),
            holding=(),
            stored=tuple(sorted(self.instance.items.items())),
            delivered=(),
        )

    def apply(self, state: State, action: str) -> State | None:
        """The successor state, or None if the action cannot be executed."""
        name, *args = action.strip("()").split()
        positions = dict(state.at)
        holding = dict(state.holding)
        stored = dict(state.stored)
        delivered = dict(state.delivered)

        if name == "move":
            robot, origin, destination = args
            if positions.get(robot) != origin or (origin, destination) not in self.connected:
                return None
            if destination in positions.values():
                return None
            positions[robot] = destination
        elif name == "pick":
            robot, item, shelf = args
            if not shelf.startswith("shelf") or positions.get(robot) != shelf:
                return None
            if stored.get(item) != shelf or robot in holding:
                return None
            del stored[item]
            holding[robot] = item
        elif name == "drop":
            robot, item, station = args
            if not station.startswith("station") or positions.get(robot) != station:
                return None
            if holding.get(robot) != item:
                return None
            del holding[robot]
            delivered[item] = station
        else:
            raise ValueError(action)

        return State(
            tuple(sorted(positions.items())),
            tuple(sorted(holding.items())),
            tuple(sorted(stored.items())),
            tuple(sorted(delivered.items())),
        )

    def is_goal(self, state: State) -> bool:
        return self.goal <= set(state.delivered)

    def verdict(self, actions: list[str]) -> str:
        """Which of 'valid', 'precondition', 'goal' describes a well-formed action list."""
        state = self.initial()
        for action in actions:
            state = self.apply(state, action)
            if state is None:
                return "precondition"
        return "valid" if self.is_goal(state) else "goal"

    def successors(self, state: State):
        positions = dict(state.at)
        holding = dict(state.holding)
        stored = dict(state.stored)
        for robot, location in positions.items():
            for origin, destination in self.connected:
                if origin == location:
                    yield f"(move {robot} {origin} {destination})"
            for item, shelf in stored.items():
                if shelf == location:
                    yield f"(pick {robot} {item} {shelf})"
            if robot in holding:
                yield f"(drop {robot} {holding[robot]} {location})"

    def shortest_plan_length(self) -> int:
        start = self.initial()
        frontier = deque([(start, 0)])
        seen = {start}
        while frontier:
            state, depth = frontier.popleft()
            if self.is_goal(state):
                return depth
            for action in self.successors(state):
                following = self.apply(state, action)
                if following is not None and following not in seen:
                    seen.add(following)
                    frontier.append((following, depth + 1))
        raise AssertionError("unsolvable problem")
