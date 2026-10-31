from PIL import Image

from warehouse_planning import DOMAIN_PATH
from warehouse_planning.animate import _frames, hardest_repaired_run, render_gif
from warehouse_planning.instances import load_split
from warehouse_planning.loop import FEEDBACK, Attempt, Run
from warehouse_planning.planner import load_optimal
from warehouse_planning.validator import validate

INSTANCE = load_split("benchmark")[0]
PLAN = load_optimal("benchmark")[INSTANCE.name]["plan"]


def attempt(text: str, locked=()) -> Attempt:
    result = validate(DOMAIN_PATH, INSTANCE.pddl, text, locked=locked)
    return Attempt(text, result.outcome.value, result.actions, result.failed_step, result.feedback, result.trace)


def repaired_run(tier: int = 1) -> Run:
    first = attempt("\n".join(PLAN[:3]) + "\n(fly robot1 a00 a10)")
    second = attempt("\n".join(PLAN[3:]), locked=PLAN[:3])
    return Run(INSTANCE.name, tier, 0, FEEDBACK, len(PLAN), [first, second])


def test_locked_steps_are_not_replayed():
    frames = _frames(INSTANCE, repaired_run())
    second_attempt = [f for f in frames if f.attempt_number == 2 and f.status == "running"]
    assert second_attempt[0].current == 2
    assert [f.current for f in second_attempt[1:]] == list(range(3, len(PLAN)))


def test_each_attempt_ends_with_its_verdict():
    statuses = [f.status for f in _frames(INSTANCE, repaired_run())]
    assert statuses.count("failed") == 1
    assert statuses[-1] == "valid"


def test_gif_has_one_frame_per_step(tmp_path):
    path = tmp_path / "plan.gif"
    run = repaired_run()
    render_gif(INSTANCE, run, path)
    with Image.open(path) as gif:
        assert gif.n_frames == len(_frames(INSTANCE, run))


def test_animation_picks_the_hardest_repair():
    easy, hard = repaired_run(tier=1), repaired_run(tier=4)
    assert hardest_repaired_run([easy, hard]) is hard
