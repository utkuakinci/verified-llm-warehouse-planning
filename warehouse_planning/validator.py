"""Turn raw LLM text into a plan, verify it with unified-planning, and explain what went wrong.

The plan is walked step by step to find the first thing that is wrong, in execution order, and to
describe the state at that point. The verdict itself comes from unified-planning's sequential plan
validator: if the walk ever disagrees with it, validation stops with an error.
"""

import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from itertools import product
from pathlib import Path

import unified_planning.shortcuts as up
from unified_planning.engines import ValidationResultStatus
from unified_planning.io import PDDLReader
from unified_planning.plans import ActionInstance, SequentialPlan

up.get_environment().credits_stream = None

ACTION_PATTERN = re.compile(r"\(\s*([A-Za-z][\w-]*)((?:\s+[A-Za-z0-9_-]+)*)\s*\)")


class Outcome(str, Enum):
    VALID = "valid"
    NO_PLAN = "no_plan"  # Nothing that looks like an action in the output.
    MALFORMED = "malformed"  # Unknown action or object, wrong argument count or type.
    PRECONDITION = "precondition"  # An action cannot be executed in the state it is reached in.
    GOAL = "goal"  # All actions execute, but some goal is not reached.


@dataclass
class Validation:
    outcome: Outcome
    actions: list[str]  # Every action the output contained, normalized.
    feedback: str  # Empty for a valid plan.
    failed_step: int | None = None  # 1-based; None unless a specific step is at fault.
    # State after 0, 1, 2, ... executed actions, as PDDL atoms. Drives the animation.
    trace: list[list[str]] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.outcome is Outcome.VALID


def valid_prefix(actions: list[str], trace: list[list[str]]) -> list[str]:
    """The leading actions that are known to execute: the trace has one state per executed action."""
    return actions[: max(len(trace) - 1, 0)]


def format_action(action: ActionInstance) -> str:
    return f"({' '.join([action.action.name, *map(str, action.actual_parameters)])})"


def _conjuncts(conditions) -> list:
    """Flatten a list of conditions: unified-planning keeps `(and a b)` as one node."""
    return [atom for condition in conditions for atom in (condition.args if condition.is_and() else (condition,))]


def _format_fluent(expression) -> str:
    """at(robot1, a00) -> (at robot1 a00)"""
    return f"({' '.join([expression.fluent().name, *map(str, expression.args)])})"


def extract_actions(text: str) -> list[tuple[str, list[str]]]:
    """Read `(name arg ...)` groups from free-form model output.

    Only the first unbroken run of actions counts as the plan: a line of prose after it ends the
    plan, so whatever a model rambles on with afterwards is not silently executed.
    """
    actions: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        matches = ACTION_PATTERN.findall(line)
        if not matches:
            is_comment = line.lstrip().startswith(";")
            if actions and not is_comment and re.search(r"[A-Za-z]{3}", line):
                break
            continue
        for name, args in matches:
            actions.append((name.lower(), args.lower().split()))
    return actions


@lru_cache(maxsize=None)
def _parse_cached(domain_path: str, problem_text: str):
    with tempfile.TemporaryDirectory() as tmp:
        problem_path = Path(tmp) / "problem.pddl"
        problem_path.write_text(problem_text, encoding="utf-8")
        return PDDLReader().parse_problem(domain_path, str(problem_path))


def parse_problem(domain_path: Path, problem_text: str):
    return _parse_cached(str(domain_path), problem_text)


def _signature(action) -> str:
    return f"({' '.join([action.name, *(f'<{p.type.name}>' for p in action.parameters)])})"


def _grammar(problem) -> str:
    return ", ".join(_signature(a) for a in problem.actions)


def _instantiate(problem, name: str, args: list[str], step: int) -> ActionInstance | str:
    """Return the ground action, or an error message saying why it does not exist."""
    written = f"({' '.join([name, *args])})"
    if not problem.has_action(name):
        return f"Step {step}: {written} is not an action. The allowed actions are {_grammar(problem)}."
    action = problem.action(name)
    if len(args) != len(action.parameters):
        return (
            f"Step {step}: {written} has {len(args)} arguments but {name} takes "
            f"{len(action.parameters)}: {_signature(action)}."
        )
    objects = []
    for arg, parameter in zip(args, action.parameters):
        if not problem.has_object(arg):
            return f"Step {step}: {written} uses {arg}, which is not an object of this problem."
        obj = problem.object(arg)
        if not parameter.type.is_compatible(obj.type):
            return (
                f"Step {step}: {written} uses {arg}, which has type {obj.type.name}, "
                f"where {name} needs type {parameter.type.name}: {_signature(action)}."
            )
        objects.append(obj)
    return ActionInstance(action, tuple(up.ObjectExp(o) for o in objects))


def _true_atoms(problem, state) -> list[str]:
    """Everything true in `state`, except the static map and `clear`, which are implied."""
    return [
        _format_fluent(expression)
        for fluent in problem.fluents
        if fluent.name not in ("connected", "clear")
        for expression in _ground(problem, fluent)
        if state.get_value(expression).is_true()
    ]


def _ground(problem, fluent):
    choices = [problem.objects(parameter.type) for parameter in fluent.signature]
    for combination in product(*choices):
        yield up.FluentExp(fluent, tuple(up.ObjectExp(o) for o in combination))


def _neighbours(problem, location: str) -> list[str]:
    connected = problem.fluent("connected")
    origin = up.ObjectExp(problem.object(location))
    return [
        destination.name
        for destination in problem.objects(problem.user_type("location"))
        if problem.initial_value(connected(origin, up.ObjectExp(destination))).is_true()
    ]


def _unsatisfied_preconditions(state, action: ActionInstance) -> list:
    """Ground and evaluate the preconditions directly.

    The simulator's get_unsatisfied_conditions raises for actions its grounder pruned as statically
    impossible, such as a move between unconnected locations, which is a common model error.
    """
    bindings = {up.ParameterExp(p): v for p, v in zip(action.action.parameters, action.actual_parameters)}
    grounded = (atom.substitute(bindings) for atom in _conjuncts(action.action.preconditions))
    return [atom for atom in grounded if not state.get_value(atom).is_true()]


def _map_hints(problem, action: ActionInstance, unsatisfied: list, atoms: list[str]) -> list[str]:
    """What the map says about a rejected action: only facts of the problem, never a route."""
    robot = str(action.actual_parameters[0])
    hints = []
    location = next((a.strip("()").split()[2] for a in atoms if a.startswith(f"(at {robot} ")), None)
    if location is not None:
        hints.append(f"From {location}, {robot} can move to: {', '.join(_neighbours(problem, location))}.")
    for condition in unsatisfied:
        if condition.fluent().name == "connected":
            target = str(condition.args[1])
            hints.append(f"{target} can only be entered from: {', '.join(_neighbours(problem, target))}.")
    return hints


def _accepted_by_validator(problem, instances: list[ActionInstance]) -> bool:
    plan = SequentialPlan(instances, problem.environment)
    with up.PlanValidator(name="sequential_plan_validator") as validator:
        return validator.validate(problem, plan).status is ValidationResultStatus.VALID


def validate(domain_path: Path, problem_text: str, llm_output: str, locked: Sequence[str] = ()) -> Validation:
    """Check model output against the problem.

    `locked` holds steps verified in an earlier round. The output continues them, and is also
    accepted if it restates them before continuing.
    """
    problem = parse_problem(domain_path, problem_text)
    kept = extract_actions("\n".join(locked))
    new = extract_actions(llm_output)
    if new[: len(kept)] == kept:
        new = new[len(kept) :]
    parsed = kept + new
    written = [f"({' '.join([name, *args])})" for name, args in parsed]

    with up.SequentialSimulator(problem) as simulator:
        state = simulator.get_initial_state()
        trace = [_true_atoms(problem, state)]
        instances: list[ActionInstance] = []

        for step, (name, args) in enumerate(parsed, start=1):
            instance = _instantiate(problem, name, args, step)
            if isinstance(instance, str):
                return Validation(Outcome.MALFORMED, written, instance, step, trace)
            if not simulator.is_applicable(state, instance):
                if _accepted_by_validator(problem, [*instances, instance]):
                    raise AssertionError("the validator accepted an inapplicable action")
                unsatisfied = _unsatisfied_preconditions(state, instance)
                missing = " ".join(_format_fluent(c) for c in unsatisfied)
                feedback = [
                    f"Step {step}: {format_action(instance)} cannot be executed.",
                    f"Unsatisfied precondition: {missing}.",
                    *_map_hints(problem, instance, unsatisfied, trace[-1]),
                ]
                return Validation(Outcome.PRECONDITION, written, "\n".join(feedback), step, trace)
            state = simulator.apply(state, instance)
            trace.append(_true_atoms(problem, state))
            instances.append(instance)

        if not new:
            feedback = f"No actions found in your answer. Write one per line, using {_grammar(problem)}."
            return Validation(Outcome.NO_PLAN, written, feedback, trace=trace)

        open_goals = [g for g in _conjuncts(problem.goals) if not state.get_value(g).is_true()]
        if (not open_goals) != _accepted_by_validator(problem, instances):
            raise AssertionError("the validator and the state walk disagree about the goal")
        if not open_goals:
            return Validation(Outcome.VALID, written, "", trace=trace)
        feedback = [
            f"All {len(instances)} steps can be executed, but the goal is not reached.",
            f"Unsatisfied goal: {' '.join(_format_fluent(g) for g in open_goals)}.",
        ]
        return Validation(Outcome.GOAL, written, "\n".join(feedback), trace=trace)
