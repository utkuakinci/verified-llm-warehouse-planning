"""Run the planning experiment.

    python -m warehouse_planning.experiment --label qwen3-4b
    python -m warehouse_planning.experiment --label qwen25-1.5b --model Qwen/Qwen2.5-1.5B-Instruct --batch-size 5
    python -m warehouse_planning.experiment --label claude --backend claude
"""

import argparse
import json

from warehouse_planning import RESULTS_DIR
from warehouse_planning.instances import load_split
from warehouse_planning.llm import DEFAULT_CLAUDE_MODEL, DEFAULT_HF_MODEL, ClaudeLLM, HuggingFaceLLM
from warehouse_planning.loop import run_experiment
from warehouse_planning.metrics import summarize
from warehouse_planning.planner import load_optimal

DEFAULT_REPLICATES = 2
DEFAULT_MAX_REPAIRS = 4
DEFAULT_TEMPERATURE = 0.3
DEFAULT_REPAIR_TEMPERATURE = 0.7
# On a 4 GB GPU the 4B model takes 2.66 GB for its weights and 146 KB per token of cache per
# sequence, so three long sequences no longer fit. The 1.5B model takes a batch of 5.
DEFAULT_BATCH_SIZE = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="results go to results/raw/<label>.json")
    parser.add_argument("--backend", choices=["hf", "claude"], default="hf")
    parser.add_argument("--model", help=f"defaults: {DEFAULT_HF_MODEL} (hf), {DEFAULT_CLAUDE_MODEL} (claude)")
    parser.add_argument("--adapter", help="path to a LoRA adapter (hf only)")
    parser.add_argument("--split", default="benchmark", choices=["benchmark", "dev", "train", "unseen"])
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--max-repairs", type=int, default=DEFAULT_MAX_REPAIRS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--repair-temperature", type=float, default=DEFAULT_REPAIR_TEMPERATURE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-tier", type=int, default=99, help="skip harder tiers, for quick trials")
    args = parser.parse_args()

    if args.backend == "claude":
        llm = ClaudeLLM(args.model or DEFAULT_CLAUDE_MODEL)
    else:
        llm = HuggingFaceLLM(args.model or DEFAULT_HF_MODEL, adapter=args.adapter, four_bit=True)

    instances = [i for i in load_split(args.split) if i.tier <= args.max_tier]
    optimal = {name: entry["length"] for name, entry in load_optimal(args.split).items()}
    out_path = RESULTS_DIR / "raw" / f"{args.label}.json"

    runs = run_experiment(
        llm,
        instances,
        optimal,
        replicates=args.replicates,
        max_repairs=args.max_repairs,
        temperature=args.temperature,
        repair_temperature=args.repair_temperature,
        batch_size=args.batch_size,
        out_path=out_path,
        log=lambda line: print(line, flush=True),
    )

    summary = summarize(runs, args.max_repairs)
    summary["model"] = llm.name
    summary["settings"] = {k: v for k, v in vars(args).items() if k != "label"}
    summary_path = RESULTS_DIR / f"{args.label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({arm: {k: v for k, v in m.items() if k.endswith("rate")} for arm, m in summary["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
