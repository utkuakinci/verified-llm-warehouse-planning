from transformers import AutoTokenizer

from warehouse_planning.finetune import planner_examples, self_examples, tokenize
from warehouse_planning.instances import load_split
from warehouse_planning.loop import FEEDBACK, Attempt, Run, save_runs
from warehouse_planning.planner import load_optimal

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN = load_split("train")


def test_only_the_plan_is_learned():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pddl, plan = planner_examples(1)[0]
    ids, labels = tokenize(tokenizer, pddl, plan)
    learned = labels[labels != -100]
    assert len(ids) == len(labels)
    assert tokenizer.decode(learned).strip() == "\n".join(plan) + tokenizer.eos_token
    prompt = tokenizer.decode(ids[labels == -100])
    assert prompt.endswith("Plan:<|im_end|>\n<|im_start|>assistant\n")


def test_planner_examples_come_from_the_train_split_only():
    examples = planner_examples(2)
    benchmark = {i.pddl for i in load_split("benchmark")}
    assert len(examples) == 40
    assert not {pddl for pddl, _ in examples} & benchmark


def test_planner_examples_respect_the_tier_limit():
    assert len(planner_examples(1)) == 20
    assert len(planner_examples(99)) == len(load_optimal("train"))


def test_self_examples_keep_valid_plans_once(tmp_path):
    instance = TRAIN[0]
    plan = load_optimal("train")[instance.name]["plan"]
    valid = Attempt("", "valid", plan, None, "", [])
    failed = Attempt("", "goal", plan[:1], None, "", [])
    runs = [
        Run(instance.name, 1, 0, FEEDBACK, len(plan), [valid]),
        Run(instance.name, 1, 1, FEEDBACK, len(plan), [failed, valid]),
        Run(TRAIN[1].name, 1, 0, FEEDBACK, len(plan), [failed]),
    ]
    path = tmp_path / "runs.json"
    save_runs(runs, path)
    assert self_examples(path) == [(instance.pddl, plan)]
