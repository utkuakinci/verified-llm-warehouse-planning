"""The generate, validate, repair loop, run over many problems at once.

Two arms share the same first attempt:

- `feedback`: steps the validator accepted are locked, and each retry shows the model the
  validator's explanation and asks only for what comes next. The full plan is verified again.
- `blind`: every retry resamples the original prompt, so it gets the same number of tries with no
  information about what went wrong. It is the control that tells a benefit of the feedback apart
  from the benefit of simply trying more than once.
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from warehouse_planning import DOMAIN_PATH
from warehouse_planning.instances import Instance
from warehouse_planning.llm import LLM
from warehouse_planning.prompts import planning_messages, repair_messages
from warehouse_planning.validator import Outcome, valid_prefix, validate

FEEDBACK = "feedback"
BLIND = "blind"
ARMS = (FEEDBACK, BLIND)

# Generation budget. Output past this length is cut, so a plan can be at most 2.5 times the optimum.
TOKENS_PER_ACTION = 14
MAX_LENGTH_RATIO = 2.5


@dataclass
class Attempt:
    output: str
    outcome: str
    actions: list[str]
    failed_step: int | None
    feedback: str
    trace: list[list[str]]


@dataclass
class Run:
    problem: str
    tier: int
    replicate: int
    arm: str
    optimal_length: int
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def solved(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].outcome == "valid"


def _chunks(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _locked_steps(run: Run) -> list[str]:
    if run.arm != FEEDBACK or not run.attempts:
        return []
    last = run.attempts[-1]
    return valid_prefix(last.actions, last.trace)


def _next_messages(run: Run, instance: Instance) -> list[dict]:
    if run.arm == BLIND or not run.attempts:
        return planning_messages(instance.pddl)
    last = run.attempts[-1]
    return repair_messages(instance.pddl, last.trace[-1], last.feedback)


def _token_budget(chunk: list[Run]) -> int:
    return int(TOKENS_PER_ACTION * MAX_LENGTH_RATIO * max(r.optimal_length for r in chunk))


def _verifier(run: Run, instance: Instance):
    """Tell generation to stop at the first wrong step: nothing after it can ever be used."""
    locked = _locked_steps(run)

    def wrong_step_reached(text: str) -> bool:
        return validate(DOMAIN_PATH, instance.pddl, text, locked=locked).outcome in (
            Outcome.MALFORMED,
            Outcome.PRECONDITION,
            Outcome.VALID,
        )

    return wrong_step_reached


def _generate(llm: LLM, chunk: list[Run], by_name: dict[str, Instance], temperature: float, seed: int) -> list[str]:
    batch = [_next_messages(run, by_name[run.problem]) for run in chunk]
    verifiers = [_verifier(run, by_name[run.problem]) for run in chunk]
    return llm.generate(batch, temperature, seed, _token_budget(chunk), verifiers)


def _record(run: Run, instance: Instance, output: str) -> None:
    result = validate(DOMAIN_PATH, instance.pddl, output, locked=_locked_steps(run))
    run.attempts.append(
        Attempt(output, result.outcome.value, result.actions, result.failed_step, result.feedback, result.trace)
    )


def save_runs(runs: list[Run], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps([asdict(r) for r in runs]), encoding="utf-8")
    temporary.replace(path)


def load_runs(path: Path) -> list[Run]:
    runs = []
    for record in json.loads(path.read_text(encoding="utf-8")):
        attempts = [Attempt(**a) for a in record.pop("attempts")]
        runs.append(Run(attempts=attempts, **record))
    return runs


def run_experiment(
    llm: LLM,
    instances: list[Instance],
    optimal_lengths: dict[str, int],
    replicates: int,
    max_repairs: int,
    temperature: float,
    repair_temperature: float,
    batch_size: int,
    out_path: Path,
    arms: tuple[str, ...] = ARMS,
    log: Callable[[str], None] = print,
) -> list[Run]:
    """Run every (problem, replicate) through both arms, saving after each batch.

    Both arms retry at `repair_temperature`: at the first attempt's lower temperature the model
    keeps writing the same plan, and a blind retry needs variety to be a fair control at all.

    If `out_path` exists the experiment resumes from it.
    """
    by_name = {i.name: i for i in instances}
    if out_path.exists():
        runs = load_runs(out_path)
    else:
        runs = [
            Run(instance.name, instance.tier, replicate, arm, optimal_lengths[instance.name])
            for arm in arms
            for replicate in range(replicates)
            for instance in instances
        ]

    # Round 0: identical for both arms, so generate once and copy it across.
    primary = [r for r in runs if r.arm == arms[0]]
    pending = [r for r in primary if not r.attempts]
    for index, chunk in enumerate(_chunks(_by_prompt_length(pending, by_name), batch_size)):
        outputs = _generate(llm, chunk, by_name, temperature, _seed(0, index))
        for run, output in zip(chunk, outputs):
            _record(run, by_name[run.problem], output)
            for twin in runs:
                if twin is not run and (twin.problem, twin.replicate) == (run.problem, run.replicate):
                    twin.attempts = list(run.attempts)
        save_runs(runs, out_path)
        log(f"round 0: {sum(r.attempts != [] for r in primary)}/{len(primary)} first attempts")

    for round_index in range(1, max_repairs + 1):
        for arm_number, arm in enumerate(arms, start=1):
            pending = [r for r in runs if r.arm == arm and not r.solved and len(r.attempts) == round_index]
            for index, chunk in enumerate(_chunks(_by_prompt_length(pending, by_name), batch_size)):
                seed = _seed(round_index * 10 + arm_number, index)
                outputs = _generate(llm, chunk, by_name, repair_temperature, seed)
                for run, output in zip(chunk, outputs):
                    _record(run, by_name[run.problem], output)
                save_runs(runs, out_path)
            solved = sum(r.solved for r in runs if r.arm == arm)
            total = sum(r.arm == arm for r in runs)
            log(f"round {round_index} {arm}: {solved}/{total} solved")
    return runs


def _seed(stage: int, chunk: int) -> int:
    return stage * 10_000 + chunk


def _by_prompt_length(runs: list[Run], by_name: dict[str, Instance]) -> list[Run]:
    """Group similar lengths together, so batches waste less memory on padding."""
    return sorted(runs, key=lambda r: (r.tier, len(by_name[r.problem].pddl), r.problem, r.replicate))
