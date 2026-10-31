"""Turn raw runs into the reported metrics.

Confidence intervals resample whole problems, not individual runs: replicates of one problem share
its difficulty, so treating them as independent would make the intervals too narrow.
"""

from collections import defaultdict
from statistics import mean, median

import numpy as np

from warehouse_planning.instances import TIERS
from warehouse_planning.loop import BLIND, FEEDBACK, Run

BOOTSTRAP_SAMPLES = 5_000


def _rate(values: list[bool]) -> float | None:
    return float(mean(values)) if values else None


def _mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def first_attempt_valid(run: Run) -> bool:
    return run.attempts[0].outcome == "valid"


def repairs_used(run: Run) -> int:
    """How many retries came after the first attempt."""
    return len(run.attempts) - 1


def length_ratio(run: Run) -> float:
    """Length of a solved run's valid plan relative to the optimal plan."""
    return len(run.attempts[-1].actions) / run.optimal_length


def goal_fraction(run: Run) -> float:
    """Most items delivered by any attempt, as a share of the items to deliver.

    A run that delivers three of four items fails like one that never moves, and this tells them apart.
    """
    delivered = max(
        (sum(atom.startswith("(delivered ") for atom in attempt.trace[-1]) for attempt in run.attempts if attempt.trace),
        default=0,
    )
    return delivered / TIERS[run.tier - 1].items


def arm_metrics(runs: list[Run]) -> dict:
    solved = [r for r in runs if r.solved]
    needed_repair = [r for r in runs if not first_attempt_valid(r)]
    repaired = [r for r in needed_repair if r.solved]
    ratios = [length_ratio(r) for r in solved]
    return {
        "runs": len(runs),
        "first_attempt_valid_rate": _rate([first_attempt_valid(r) for r in runs]),
        "final_valid_rate": _rate([r.solved for r in runs]),
        "mean_goal_fraction": _mean([goal_fraction(r) for r in runs]),
        "repair_success_rate": _rate([r.solved for r in needed_repair]),
        "mean_repairs_all_solved": _mean([repairs_used(r) for r in solved]),
        "mean_repairs_when_repaired": _mean([repairs_used(r) for r in repaired]),
        "mean_length_ratio": _mean(ratios),
        "median_length_ratio": float(median(ratios)) if ratios else None,
        "optimal_plan_rate": _rate([r == 1.0 for r in ratios]),
    }


def _problem_rates(runs: list[Run], score) -> dict[str, float]:
    by_problem: dict[str, list[bool]] = defaultdict(list)
    for run in runs:
        by_problem[run.problem].append(score(run))
    return {problem: float(mean(values)) for problem, values in by_problem.items()}


def _interval(samples: np.ndarray) -> list[float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    return [float(low), float(high)]


def bootstrap_interval(runs: list[Run], score, rng: np.random.Generator) -> list[float]:
    rates = np.array(list(_problem_rates(runs, score).values()))
    picks = rng.integers(0, len(rates), size=(BOOTSTRAP_SAMPLES, len(rates)))
    return _interval(rates[picks].mean(axis=1))


def bootstrap_difference(a: list[Run], b: list[Run], score, rng: np.random.Generator) -> dict:
    """Paired difference in success rate between two arms, on the same problems."""
    rates_a, rates_b = _problem_rates(a, score), _problem_rates(b, score)
    problems = sorted(rates_a)
    differences = np.array([rates_a[p] - rates_b[p] for p in problems])
    picks = rng.integers(0, len(differences), size=(BOOTSTRAP_SAMPLES, len(differences)))
    return {"difference": float(differences.mean()), "interval": _interval(differences[picks].mean(axis=1))}


def cumulative_success(runs: list[Run], max_repairs: int) -> list[float]:
    """Share of runs solved within k+1 attempts, for k = 0..max_repairs."""
    return [mean(r.solved and repairs_used(r) <= k for r in runs) for k in range(max_repairs + 1)]


def outcome_shares(runs: list[Run], max_repairs: int) -> list[dict[str, float]]:
    """For each attempt number, the share of each outcome among runs that were still unsolved."""
    shares = []
    for index in range(max_repairs + 1):
        outcomes = [r.attempts[index].outcome for r in runs if len(r.attempts) > index]
        if outcomes:
            shares.append({o: outcomes.count(o) / len(outcomes) for o in set(outcomes)} | {"_n": len(outcomes)})
    return shares


def summarize(runs: list[Run], max_repairs: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    summary: dict = {"max_repairs": max_repairs, "arms": {}, "tiers": {}}
    arms = sorted({r.arm for r in runs})
    by_arm = {arm: [r for r in runs if r.arm == arm] for arm in arms}

    for arm, arm_runs in by_arm.items():
        metrics = arm_metrics(arm_runs)
        metrics["final_valid_interval"] = bootstrap_interval(arm_runs, lambda r: r.solved, rng)
        metrics["first_attempt_valid_interval"] = bootstrap_interval(arm_runs, first_attempt_valid, rng)
        metrics["cumulative_success"] = cumulative_success(arm_runs, max_repairs)
        metrics["outcomes_by_attempt"] = outcome_shares(arm_runs, max_repairs)
        summary["arms"][arm] = metrics

    if FEEDBACK in by_arm and BLIND in by_arm:
        summary["feedback_minus_blind"] = bootstrap_difference(
            by_arm[FEEDBACK], by_arm[BLIND], lambda r: r.solved, rng
        )

    for tier in sorted({r.tier for r in runs}):
        tier_runs = [r for r in runs if r.tier == tier]
        entry = {"mean_optimal_length": float(mean(r.optimal_length for r in tier_runs))}
        for arm, arm_runs in by_arm.items():
            selected = [r for r in arm_runs if r.tier == tier]
            entry[arm] = arm_metrics(selected)
            entry[arm]["final_valid_interval"] = bootstrap_interval(selected, lambda r: r.solved, rng)
            entry[arm]["first_attempt_valid_interval"] = bootstrap_interval(selected, first_attempt_valid, rng)
        summary["tiers"][str(tier)] = entry
    return summary
