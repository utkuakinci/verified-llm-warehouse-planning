from warehouse_planning.instances import load_split
from warehouse_planning.loop import BLIND, FEEDBACK, load_runs, run_experiment
from warehouse_planning.planner import load_optimal

INSTANCE = load_split("benchmark")[0]
OPTIMAL_PLAN = load_optimal("benchmark")[INSTANCE.name]["plan"]
GOOD = "\n".join(OPTIMAL_PLAN)
BAD = "(fly robot1 a00 a10)"
OPTIMAL = {INSTANCE.name: len(OPTIMAL_PLAN)}
REPAIR_MARKER = "last attempt from this state was rejected"


class ScriptedLLM:
    """Answers BAD until it sees validator feedback, or until it runs out of scripted answers."""

    name = "scripted"

    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.calls = []

    def generate(self, batch, temperature, seed, max_new_tokens, verifiers=None):
        self.calls.append(batch)
        self.max_new_tokens = max_new_tokens
        self.verifiers = verifiers
        outputs = []
        for messages in batch:
            if self.answers:
                outputs.append(self.answers.pop(0))
            else:
                outputs.append(GOOD if REPAIR_MARKER in messages[-1]["content"] else BAD)
        return outputs


def experiment(llm, path, max_repairs=2, replicates=1, arms=(FEEDBACK, BLIND)):
    return run_experiment(
        llm,
        [INSTANCE],
        OPTIMAL,
        replicates=replicates,
        max_repairs=max_repairs,
        temperature=0.5,
        repair_temperature=0.5,
        batch_size=4,
        out_path=path,
        arms=arms,
        log=lambda _: None,
    )


def by_arm(runs, arm):
    return next(r for r in runs if r.arm == arm)


def test_feedback_arm_repairs_and_stops(tmp_path):
    runs = experiment(ScriptedLLM(), tmp_path / "runs.json")
    feedback = by_arm(runs, FEEDBACK)
    assert feedback.solved
    assert [a.outcome for a in feedback.attempts] == ["malformed", "valid"]


def test_blind_arm_never_sees_the_error(tmp_path):
    llm = ScriptedLLM()
    runs = experiment(llm, tmp_path / "runs.json")
    blind = by_arm(runs, BLIND)
    assert not blind.solved
    assert len(blind.attempts) == 3
    conversations = [messages for call in llm.calls for messages in call]
    assert sum(REPAIR_MARKER in messages[-1]["content"] for messages in conversations) == 1


def test_arms_share_the_first_attempt_and_it_is_generated_once(tmp_path):
    llm = ScriptedLLM([GOOD])
    runs = experiment(llm, tmp_path / "runs.json")
    assert by_arm(runs, FEEDBACK).attempts[0] == by_arm(runs, BLIND).attempts[0]
    assert len(llm.calls) == 1


def test_repair_prompt_explains_the_failure(tmp_path):
    llm = ScriptedLLM()
    experiment(llm, tmp_path / "runs.json", arms=(FEEDBACK,))
    assert "(fly robot1 a00 a10) is not an action" in llm.calls[1][0][-1]["content"]


def test_verified_steps_stay_locked_across_rounds(tmp_path):
    """The model only writes the rest, and the plan is checked as a whole."""
    first_half, second_half = OPTIMAL_PLAN[:3], OPTIMAL_PLAN[3:]
    answers = ["\n".join(first_half) + "\n" + BAD, "\n".join(second_half)]
    runs = experiment(ScriptedLLM(answers), tmp_path / "runs.json", arms=(FEEDBACK,))
    assert runs[0].solved
    assert runs[0].attempts[0].failed_step == 4
    assert runs[0].attempts[1].actions == OPTIMAL_PLAN


def test_model_may_restate_locked_steps(tmp_path):
    answers = ["\n".join(OPTIMAL_PLAN[:3]) + "\n" + BAD, GOOD]
    runs = experiment(ScriptedLLM(answers), tmp_path / "runs.json", arms=(FEEDBACK,))
    assert runs[0].solved
    assert runs[0].attempts[1].actions == OPTIMAL_PLAN


def test_empty_answer_keeps_the_verified_steps(tmp_path):
    answers = ["\n".join(OPTIMAL_PLAN[:3]) + "\n" + BAD, "I give up.", "\n".join(OPTIMAL_PLAN[3:])]
    runs = experiment(ScriptedLLM(answers), tmp_path / "runs.json", arms=(FEEDBACK,))
    assert [a.outcome for a in runs[0].attempts] == ["malformed", "no_plan", "valid"]
    assert runs[0].attempts[-1].actions == OPTIMAL_PLAN


def test_blind_arm_never_locks_steps(tmp_path):
    answers = ["\n".join(OPTIMAL_PLAN[:3]) + "\n" + BAD, "\n".join(OPTIMAL_PLAN[3:])]
    runs = experiment(ScriptedLLM(answers), tmp_path / "runs.json", max_repairs=1, arms=(BLIND,))
    assert not runs[0].solved
    assert runs[0].attempts[1].actions == OPTIMAL_PLAN[3:]


def test_token_budget_scales_with_the_problem(tmp_path):
    llm = ScriptedLLM([GOOD])
    experiment(llm, tmp_path / "runs.json", arms=(FEEDBACK,))
    assert llm.max_new_tokens == int(14 * 2.5 * len(OPTIMAL_PLAN))


def test_verifier_stops_at_the_first_wrong_step_only(tmp_path):
    llm = ScriptedLLM([GOOD])
    experiment(llm, tmp_path / "runs.json", arms=(FEEDBACK,))
    verifier = llm.verifiers[0]
    assert verifier(GOOD)
    assert verifier(BAD)
    assert not verifier("\n".join(OPTIMAL_PLAN[:-1]))
    assert not verifier("")


def test_repairs_are_bounded(tmp_path):
    runs = experiment(ScriptedLLM([BAD] * 20), tmp_path / "runs.json", max_repairs=3, arms=(FEEDBACK,))
    assert len(runs[0].attempts) == 4
    assert not runs[0].solved


def test_resume_finishes_without_repeating_work(tmp_path):
    path = tmp_path / "runs.json"
    experiment(ScriptedLLM([BAD] * 20), path, max_repairs=1, arms=(FEEDBACK,))
    assert len(load_runs(path)[0].attempts) == 2

    llm = ScriptedLLM([GOOD])
    runs = experiment(llm, path, max_repairs=2, arms=(FEEDBACK,))
    assert len(runs[0].attempts) == 3
    assert runs[0].solved
    assert len(llm.calls) == 1
