"""Fine-tune the model with LoRA on verified plans.

Every example comes from the train split, whose tasks never appear in the benchmark. Two sources:

- `self`: valid plans the model produced itself on the train split, the actual self-improvement loop.
- `planner`: Fast Downward's optimal plans for the train split, a teacher that also covers the
  problems the model cannot solve yet.

    python -m warehouse_planning.finetune --source planner --model Qwen/Qwen2.5-1.5B-Instruct --out adapters/planner
    python -m warehouse_planning.experiment --label lora --model Qwen/Qwen2.5-1.5B-Instruct --adapter adapters/planner
"""

import argparse
import json
import random
from pathlib import Path

import warehouse_planning  # noqa: F401  Sets up the Windows DLL path before torch loads.
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from warehouse_planning.instances import load_split
from warehouse_planning.llm import DEFAULT_HF_MODEL
from warehouse_planning.loop import load_runs
from warehouse_planning.planner import load_optimal
from warehouse_planning.prompts import planning_messages

LORA_RANK = 16
LEARNING_RATE = 2e-4
GRADIENT_ACCUMULATION = 4
MAX_SEQUENCE_TOKENS = 2600


def self_examples(runs_path: Path) -> list[tuple[str, list[str]]]:
    """(problem PDDL, plan) for every distinct valid plan the model found on the train split."""
    pddl = {i.name: i.pddl for i in load_split("train")}
    examples = {}
    for run in load_runs(runs_path):
        if run.solved:
            examples[run.problem, tuple(run.attempts[-1].actions)] = pddl[run.problem]
    return [(text, list(plan)) for (_, plan), text in examples.items()]


def planner_examples(max_tier: int) -> list[tuple[str, list[str]]]:
    optimal = load_optimal("train")
    return [(i.pddl, optimal[i.name]["plan"]) for i in load_split("train") if i.tier <= max_tier and i.name in optimal]


def tokenize(tokenizer, problem_pddl: str, plan: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids and labels. Only the plan is learned, so the prompt's labels are masked."""
    prompt = tokenizer.apply_chat_template(planning_messages(problem_pddl), tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    answer = "\n".join(plan) + tokenizer.eos_token
    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[0]
    ids = torch.cat([prompt_ids, answer_ids])
    labels = ids.clone()
    labels[: len(prompt_ids)] = -100
    return ids, labels


def train(model_id: str, examples: list[tuple[str, list[str]]], out: Path, epochs: int, seed: int) -> list[float]:
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=2 * LORA_RANK,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    data = [tokenize(tokenizer, pddl, plan) for pddl, plan in examples]
    data = [pair for pair in data if len(pair[0]) <= MAX_SEQUENCE_TOKENS]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE)
    order = random.Random(seed)
    epoch_losses = []
    model.train()
    for epoch in range(epochs):
        order.shuffle(data)
        total = 0.0
        for index, (ids, labels) in enumerate(data, start=1):
            loss = model(input_ids=ids[None].to(model.device), labels=labels[None].to(model.device)).loss
            (loss / GRADIENT_ACCUMULATION).backward()
            total += loss.item()
            if index % GRADIENT_ACCUMULATION == 0 or index == len(data):
                optimizer.step()
                optimizer.zero_grad()
        epoch_losses.append(total / len(data))
        print(f"epoch {epoch + 1}/{epochs}: loss {epoch_losses[-1]:.4f}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    return epoch_losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["self", "planner"], required=True)
    parser.add_argument("--runs", type=Path, help="experiment on the train split (--source self)")
    parser.add_argument("--max-tier", type=int, default=99, help="planner examples up to this tier")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    examples = self_examples(args.runs) if args.source == "self" else planner_examples(args.max_tier)
    if not examples:
        raise SystemExit("no verified plans to train on")
    print(f"{len(examples)} verified plans from source '{args.source}'", flush=True)
    losses = train(args.model, examples, args.out, args.epochs, args.seed)
    record = {"source": args.source, "examples": len(examples), "epochs": args.epochs, "losses": losses}
    (args.out / "training.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
