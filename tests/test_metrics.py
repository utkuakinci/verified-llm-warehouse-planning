from warehouse_planning.loop import BLIND, FEEDBACK, Attempt, Run
from warehouse_planning.metrics import arm_metrics, cumulative_success, goal_fraction, outcome_shares, summarize


def attempt(outcome: str, actions: int = 0) -> Attempt:
    return Attempt("", outcome, ["(x)"] * actions, None, "", [])


def run(problem="p1", tier=1, arm=FEEDBACK, attempts=(), replicate=0, optimal=10):
    return Run(problem, tier, replicate, arm, optimal, list(attempts))


FIRST_TRY = run(attempts=[attempt("valid", 10)])
REPAIRED_TWICE = run(attempts=[attempt("goal"), attempt("precondition"), attempt("valid", 15)])
NEVER = run(attempts=[attempt("goal")] * 5)


def test_rates_separate_first_attempt_from_final():
    metrics = arm_metrics([FIRST_TRY, REPAIRED_TWICE, NEVER, NEVER])
    assert metrics["first_attempt_valid_rate"] == 0.25
    assert metrics["final_valid_rate"] == 0.5
    assert metrics["repair_success_rate"] == 1 / 3


def test_repair_rounds_count_only_solved_runs():
    metrics = arm_metrics([FIRST_TRY, REPAIRED_TWICE, NEVER])
    assert metrics["mean_repairs_all_solved"] == 1.0
    assert metrics["mean_repairs_when_repaired"] == 2.0


def test_length_ratio_uses_valid_plans_against_the_optimum():
    metrics = arm_metrics([FIRST_TRY, REPAIRED_TWICE, NEVER])
    assert metrics["mean_length_ratio"] == 1.25
    assert metrics["optimal_plan_rate"] == 0.5


def test_metrics_with_nothing_solved_are_empty_not_zero():
    metrics = arm_metrics([NEVER])
    assert metrics["final_valid_rate"] == 0
    assert metrics["mean_length_ratio"] is None
    assert metrics["mean_repairs_all_solved"] is None


def test_goal_fraction_counts_the_best_attempt():
    one_item = Attempt("", "goal", [], None, "", [["(delivered item1 station1)"]])
    two_items = Attempt("", "goal", [], None, "", [["(delivered item1 station1)", "(delivered item2 station1)"]])
    nothing = Attempt("", "no_plan", [], None, "", [[]])
    assert goal_fraction(run(tier=2, attempts=[one_item, two_items, nothing])) == 1.0
    assert goal_fraction(run(tier=2, attempts=[one_item])) == 0.5
    assert goal_fraction(run(tier=2, attempts=[nothing])) == 0.0


def test_goal_fraction_without_any_trace_is_zero():
    assert goal_fraction(run(tier=1, attempts=[attempt("no_plan")])) == 0.0


def test_cumulative_success_grows_with_attempts():
    assert cumulative_success([FIRST_TRY, REPAIRED_TWICE, NEVER, NEVER], 4) == [0.25, 0.25, 0.5, 0.5, 0.5]


def test_outcome_shares_only_count_runs_still_going():
    shares = outcome_shares([FIRST_TRY, REPAIRED_TWICE], 2)
    assert shares[0]["_n"] == 2
    assert shares[1] == {"precondition": 1.0, "_n": 1}


def test_interval_resamples_problems_not_runs():
    """Ten replicates of one always-solved problem must not narrow the uncertainty."""
    solved = [run(problem="easy", replicate=i, attempts=[attempt("valid", 10)]) for i in range(10)]
    failed = [run(problem="hard", replicate=i, attempts=[attempt("goal")]) for i in range(10)]
    low, high = summarize(solved + failed, 0)["arms"][FEEDBACK]["final_valid_interval"]
    assert low == 0.0
    assert high == 1.0


def test_summary_compares_arms_on_the_same_problems():
    runs = []
    for index in range(20):
        name = f"p{index}"
        runs.append(run(problem=name, arm=FEEDBACK, attempts=[attempt("goal"), attempt("valid", 10)]))
        runs.append(run(problem=name, arm=BLIND, attempts=[attempt("goal"), attempt("goal")]))
    comparison = summarize(runs, 1)["feedback_minus_blind"]
    assert comparison["difference"] == 1.0
    assert comparison["interval"] == [1.0, 1.0]
