"""Build the charts and REPORT.md from a finished experiment.

    python -m warehouse_planning.report --label qwen3-4b
    python -m warehouse_planning.report --label qwen3-4b --name "Qwen3-4B" --compare qwen25-1.5b="Qwen2.5-1.5B"
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from warehouse_planning import RESULTS_DIR
from warehouse_planning.instances import TIERS, load_split
from warehouse_planning.loop import BLIND, FEEDBACK, Run, load_runs
from warehouse_planning.metrics import first_attempt_valid

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# Validated with the dataviz palette checker. Each arm keeps its color across all charts.
FEEDBACK_COLOR = "#2a78d6"
BLIND_COLOR = "#eb6834"
FIRST_COLOR = "#1baf7a"
MODEL_COLORS = (FEEDBACK_COLOR, BLIND_COLOR, FIRST_COLOR)
GOOD = "#0ca30c"
# From "got nowhere" to "almost there": one hue, darker is closer to a valid plan.
OUTCOME_ORDER = ("no_plan", "malformed", "precondition", "goal", "valid")
OUTCOME_COLORS = dict(zip(OUTCOME_ORDER, ("#86b6ef", "#5598e7", "#2a78d6", "#184f95", GOOD)))
OUTCOME_LABELS = {
    "no_plan": "No plan found",
    "malformed": "Malformed action",
    "precondition": "Precondition failed",
    "goal": "Goal not reached",
    "valid": "Valid",
}
ARM_LABELS = {FEEDBACK: "Validator feedback", BLIND: "Blind retries"}
# Series often coincide on low tiers. A small sideways offset keeps all of them visible.
DODGE = 0.07


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "xtick.color": SECONDARY,
            "ytick.color": SECONDARY,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": HAIRLINE,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.labelcolor": SECONDARY,
        }
    )


def _percent(values) -> list[float]:
    return [100 * v for v in values]


def _tier_labels() -> list[str]:
    return [f"{t.level}\n{t.robots}R {t.items}I" for t in TIERS]


def _line(ax, x, y, color, label, interval=None, zorder=3) -> None:
    if interval is not None:
        low, high = zip(*interval)
        ax.fill_between(x, _percent(low), _percent(high), color=color, alpha=0.10, linewidth=0, zorder=1)
    ax.plot(x, _percent(y), color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=label, zorder=zorder, solid_capstyle="round")


def _finish_percent_axis(ax) -> None:
    ax.set_ylim(-3, 103)
    ax.set_yticks(range(0, 101, 25))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 25)])
    ax.grid(axis="x", visible=False)


def chart_success_by_tier(summary: dict, path) -> None:
    tiers = sorted(summary["tiers"], key=int)
    x = list(range(len(tiers)))
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    first = [summary["tiers"][t][FEEDBACK]["first_attempt_valid_rate"] for t in tiers]
    _line(ax, [v - DODGE for v in x], first, FIRST_COLOR, "First attempt")
    for arm, color, offset, label in (
        (BLIND, BLIND_COLOR, 0, "Blind"),
        (FEEDBACK, FEEDBACK_COLOR, DODGE, "Validator feedback"),
    ):
        rates = [summary["tiers"][t][arm]["final_valid_rate"] for t in tiers]
        intervals = [summary["tiers"][t][arm]["final_valid_interval"] for t in tiers]
        positions = [v + offset for v in x]
        _line(ax, positions, rates, color, f"{label}, up to {summary['max_repairs']} retries", intervals)
    ax.set_xticks(x)
    ax.set_xticklabels(_tier_labels()[: len(tiers)])
    ax.set_xlabel("Difficulty tier (R robots, I items)")
    _finish_percent_axis(ax)
    ax.set_title("Valid plans by difficulty")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_success_by_attempt(summary: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    attempts = list(range(1, summary["max_repairs"] + 2))
    for arm, color, offset in ((BLIND, BLIND_COLOR, -DODGE / 2), (FEEDBACK, FEEDBACK_COLOR, DODGE / 2)):
        series = summary["arms"][arm]["cumulative_success"]
        _line(ax, [a + offset for a in attempts], series, color, ARM_LABELS[arm])
        ax.annotate(f"{100 * series[-1]:.1f}%", (attempts[-1] + offset, 100 * series[-1]), xytext=(9, 0),
                    textcoords="offset points", va="center", color=SECONDARY)
    ax.set_xticks(attempts)
    ax.set_xlim(0.8, attempts[-1] + 0.35)
    ax.set_xlabel("Attempts used")
    ax.set_ylim(bottom=-3)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="x", visible=False)
    ax.set_title("Share of runs solved within k attempts")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_plan_length(runs: list[Run], path) -> None:
    solved = [r for r in runs if r.arm == FEEDBACK and r.solved]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    top = max([len(r.attempts[-1].actions) for r in solved] + [r.optimal_length for r in solved]) * 1.08
    ax.plot([0, top], [0, top], color=AXIS, linewidth=1, zorder=1)
    ax.annotate("Optimal", (top * 0.82, top * 0.82), xytext=(5, -13), textcoords="offset points", color=MUTED)
    groups = (("Valid on the first attempt", True, FIRST_COLOR), ("Repaired with feedback", False, FEEDBACK_COLOR))
    rng = np.random.default_rng(0)
    for label, on_first, color in groups:
        group = [r for r in solved if first_attempt_valid(r) == on_first]
        if not group:
            continue
        # Runs often coincide on integer lengths, so a small jitter keeps each dot visible.
        x = np.array([r.optimal_length for r in group]) + rng.uniform(-0.2, 0.2, len(group))
        y = np.array([len(r.attempts[-1].actions) for r in group]) + rng.uniform(-0.2, 0.2, len(group))
        ax.scatter(x, y, s=46, color=color, edgecolor=SURFACE, linewidth=1.2, label=label, zorder=3)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Optimal plan length (Fast Downward)")
    ax.set_ylabel("Length of the LLM's valid plan")
    ax.set_title("Plan length against the optimum")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_outcomes(summary: dict, path) -> None:
    shares = summary["arms"][FEEDBACK]["outcomes_by_attempt"]
    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(shares) + 1.9))
    rows = list(range(len(shares)))
    left = np.zeros(len(shares))
    for outcome in OUTCOME_ORDER:
        values = np.array([100 * row.get(outcome, 0) for row in shares])
        ax.barh(rows, values, left=left, height=0.56, color=OUTCOME_COLORS[outcome], edgecolor=SURFACE, linewidth=2,
                label=OUTCOME_LABELS[outcome])
        left += values
    ax.set_yticks(rows)
    ax.set_yticklabels([f"Attempt {i + 1}\nn = {row['_n']}" for i, row in enumerate(shares)])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("How plans fail, among runs still unsolved")
    ax.legend(loc="upper center", bbox_to_anchor=(0.46, -0.13), ncols=5, fontsize=8, handlelength=1, columnspacing=1.1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_models(models: list[tuple[str, dict]], path) -> None:
    tiers = sorted(models[0][1]["tiers"], key=int)
    x = list(range(len(tiers)))
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for index, (name, summary) in enumerate(models):
        offset = DODGE * (index - (len(models) - 1) / 2)
        rates = [summary["tiers"][t][FEEDBACK]["final_valid_rate"] for t in tiers]
        intervals = [summary["tiers"][t][FEEDBACK]["final_valid_interval"] for t in tiers]
        _line(ax, [v + offset for v in x], rates, MODEL_COLORS[index], name, intervals)
    ax.set_xticks(x)
    ax.set_xticklabels(_tier_labels()[: len(tiers)])
    ax.set_xlabel("Difficulty tier (R robots, I items)")
    _finish_percent_axis(ax)
    ax.set_title("Valid plans by model, with validator feedback")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _span(interval: list[float]) -> str:
    return f"{100 * interval[0]:.1f} to {100 * interval[1]:.1f}%"


def _headline_rows(summary: dict) -> list[str]:
    arms = summary["arms"]
    feedback, blind = arms[FEEDBACK], arms[BLIND]
    rows = [
        "| Metric | Validator feedback | Blind retries |",
        "|---|---:|---:|",
        f"| Runs | {feedback['runs']} | {blind['runs']} |",
        f"| Valid on the first attempt | {_pct(feedback['first_attempt_valid_rate'])} | {_pct(blind['first_attempt_valid_rate'])} |",
        f"| 95% interval | {_span(feedback['first_attempt_valid_interval'])} | {_span(blind['first_attempt_valid_interval'])} |",
        f"| **Valid after up to {summary['max_repairs']} retries** | **{_pct(feedback['final_valid_rate'])}** | **{_pct(blind['final_valid_rate'])}** |",
        f"| 95% interval | {_span(feedback['final_valid_interval'])} | {_span(blind['final_valid_interval'])} |",
        f"| Items delivered, share of all items | {_pct(feedback['mean_goal_fraction'])} | {_pct(blind['mean_goal_fraction'])} |",
        f"| Failed first attempts later fixed | {_pct(feedback['repair_success_rate'])} | {_pct(blind['repair_success_rate'])} |",
        f"| Mean retries, all solved runs | {_num(feedback['mean_repairs_all_solved'])} | {_num(blind['mean_repairs_all_solved'])} |",
        f"| Mean retries, runs that needed one | {_num(feedback['mean_repairs_when_repaired'])} | {_num(blind['mean_repairs_when_repaired'])} |",
        f"| Plan length / optimal, mean | {_num(feedback['mean_length_ratio'])} | {_num(blind['mean_length_ratio'])} |",
        f"| Plan length / optimal, median | {_num(feedback['median_length_ratio'])} | {_num(blind['median_length_ratio'])} |",
        f"| Valid plans that are optimal | {_pct(feedback['optimal_plan_rate'])} | {_pct(blind['optimal_plan_rate'])} |",
    ]
    return rows


def _tier_rows(summary: dict) -> list[str]:
    rows = [
        "| Tier | Robots, items | Mean optimal length | First attempt | Blind retries | Validator feedback | Items delivered, blind | Items delivered, feedback |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in TIERS:
        entry = summary["tiers"].get(str(tier.level))
        if entry is None:
            continue
        feedback, blind = entry[FEEDBACK], entry[BLIND]
        rows.append(
            f"| {tier.level} | {tier.robots}, {tier.items} | {entry['mean_optimal_length']:.1f} "
            f"| {_pct(feedback['first_attempt_valid_rate'])} | {_pct(blind['final_valid_rate'])} "
            f"| {_pct(feedback['final_valid_rate'])} | {_pct(blind['mean_goal_fraction'])} | {_pct(feedback['mean_goal_fraction'])} |"
        )
    return rows


def _outcome_rows(summary: dict) -> list[str]:
    rows = [
        "| Attempt | Runs | " + " | ".join(OUTCOME_LABELS[o] for o in OUTCOME_ORDER) + " |",
        "|---|---:|" + "---:|" * len(OUTCOME_ORDER),
    ]
    for index, row in enumerate(summary["arms"][FEEDBACK]["outcomes_by_attempt"], start=1):
        cells = " | ".join(_pct(row.get(o, 0)) for o in OUTCOME_ORDER)
        rows.append(f"| {index} | {row['_n']} | {cells} |")
    return rows


def _model_rows(models: list[tuple[str, dict]]) -> list[str]:
    rows = [
        "| Model | Runs | First attempt | After retries, feedback | After retries, blind | Items delivered, feedback | Length / optimal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in models:
        feedback, blind = summary["arms"][FEEDBACK], summary["arms"][BLIND]
        rows.append(
            f"| {name} | {feedback['runs']} | {_pct(feedback['first_attempt_valid_rate'])} "
            f"| {_pct(feedback['final_valid_rate'])} | {_pct(blind['final_valid_rate'])} "
            f"| {_pct(feedback['mean_goal_fraction'])} | {_num(feedback['mean_length_ratio'])} |"
        )
    return rows


def build(label: str, name: str | None = None, compare: list[tuple[str, str]] | None = None) -> None:
    _style()
    summary = json.loads((RESULTS_DIR / f"{label}_summary.json").read_text(encoding="utf-8"))
    runs = load_runs(RESULTS_DIR / "raw" / f"{label}.json")
    figures = RESULTS_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    chart_success_by_tier(summary, figures / f"{label}_success_by_tier.png")
    chart_success_by_attempt(summary, figures / f"{label}_success_by_attempt.png")
    has_feedback_plans = any(r.solved and r.arm == FEEDBACK for r in runs)
    if has_feedback_plans:
        chart_plan_length(runs, figures / f"{label}_plan_length.png")
    chart_outcomes(summary, figures / f"{label}_outcomes.png")

    difference = summary["feedback_minus_blind"]
    lines = [
        f"# Results: {summary['model']}",
        "",
        "Generated by `python -m warehouse_planning.report`. Do not edit by hand.",
        "",
        f"{len({r.problem for r in runs})} problems, {summary['settings']['replicates']} replicates each, "
        f"temperature {summary['settings']['temperature']}, up to {summary['max_repairs']} retries. "
        "Intervals are 95% bootstraps over problems.",
        "",
        *_headline_rows(summary),
        "",
        f"Feedback minus blind retries: {100 * difference['difference']:+.1f} points "
        f"(95% interval {100 * difference['interval'][0]:+.1f} to {100 * difference['interval'][1]:+.1f}).",
        "",
        "## By difficulty",
        "",
        f"![Valid plans by difficulty](figures/{label}_success_by_tier.png)",
        "",
        *_tier_rows(summary),
        "",
        "## By attempt",
        "",
        f"![Share of runs solved within k attempts](figures/{label}_success_by_attempt.png)",
        "",
        f"![How plans fail](figures/{label}_outcomes.png)",
        "",
        *_outcome_rows(summary),
        "",
    ]
    if has_feedback_plans:
        lines += ["## Plan quality", "", f"![Plan length against the optimum](figures/{label}_plan_length.png)", ""]

    if compare:
        models = [(name or summary["model"], summary)]
        for other_label, other_name in compare:
            models.append((other_name, json.loads((RESULTS_DIR / f"{other_label}_summary.json").read_text(encoding="utf-8"))))
        chart_models(models, figures / f"{label}_models.png")
        lines += ["## By model", "", f"![Valid plans by model](figures/{label}_models.png)", "", *_model_rows(models), ""]

    (RESULTS_DIR / f"REPORT_{label}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS_DIR / f'REPORT_{label}.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--name", help="how to call this model in comparisons")
    parser.add_argument("--compare", action="append", default=[], metavar="LABEL=NAME", help="another finished run")
    args = parser.parse_args()
    build(args.label, args.name, [tuple(item.split("=", 1)) for item in args.compare])


if __name__ == "__main__":
    main()
