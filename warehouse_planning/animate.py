"""Render a run as a GIF: the warehouse, the plan, and where and why the validator stopped it.

Each frame draws the validator's own state trace, so what is on screen is what was verified.

    python -m warehouse_planning.animate --label qwen3-4b --problem p011 --replicate 0
    python -m warehouse_planning.animate --label qwen3-4b        # hardest repaired run
"""

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from warehouse_planning import RESULTS_DIR
from warehouse_planning.instances import Instance, load_split
from warehouse_planning.loop import FEEDBACK, Attempt, Run, load_runs
from warehouse_planning.validator import valid_prefix

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
AXIS = "#c3c2b7"
ROBOT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
GOOD = "#0ca30c"
GOOD_TEXT = "#006300"
CRITICAL = "#d03b3b"
AISLE_FILL = "#f4f3f0"
SHELF_FILL = "#c3c2b7"
ITEM_FILL = "#52514e"

STEP_MS = 420
HOLD_FAILURE_MS = 4200
HOLD_SUCCESS_MS = 3600
PLAN_WINDOW = 15


@dataclass
class Frame:
    attempt_number: int
    atoms: list[str]
    current: int  # 0-based index into the actions; -1 before the first action.
    status: str  # "running", "failed" or "valid"
    duration: int


def _initial_atoms(instance: Instance) -> list[str]:
    return [f"(at {r} {loc})" for r, loc in instance.robots.items()] + [
        f"(stored {i} {shelf})" for i, shelf in instance.items.items()
    ]


def _parse(atoms: list[str]) -> dict[str, list[list[str]]]:
    parsed: dict[str, list[list[str]]] = {}
    for atom in atoms:
        predicate, *args = atom.strip("()").split()
        parsed.setdefault(predicate, []).append(args)
    return parsed


def _short(name: str) -> str:
    return name.replace("robot", "R").replace("item", "i").replace("station", "T").replace("shelf", "S")


def _frames(instance: Instance, run: Run) -> list[Frame]:
    """One frame per executed step. Steps locked in an earlier round are not replayed."""
    frames = []
    initial = _initial_atoms(instance)
    locked = 0
    for number, attempt in enumerate(run.attempts, start=1):
        trace = attempt.trace or [initial]
        frames.append(Frame(number, trace[min(locked, len(trace) - 1)], locked - 1, "running", STEP_MS * 2))
        for step, atoms in enumerate(trace[1:]):
            if step >= locked:
                frames.append(Frame(number, atoms, step, "running", STEP_MS))
        if run.arm == FEEDBACK:
            locked = len(valid_prefix(attempt.actions, attempt.trace))
        if attempt.outcome == "valid":
            frames.append(Frame(number, trace[-1], len(attempt.actions) - 1, "valid", HOLD_SUCCESS_MS))
        else:
            failed = attempt.failed_step - 1 if attempt.failed_step else len(attempt.actions) - 1
            frames.append(Frame(number, trace[-1], failed, "failed", HOLD_FAILURE_MS))
    return frames


def _draw_warehouse(ax, instance: Instance, atoms: list[str], failed_action: str | None) -> None:
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")
    columns = [xy[0] for xy in instance.coords.values()]
    rows = [xy[1] for xy in instance.coords.values()]
    ax.set_xlim(min(columns) - 0.7, max(columns) + 0.7)
    ax.set_ylim(max(rows) + 0.7, min(rows) - 0.7)

    state = _parse(atoms)
    for location, (x, y) in instance.coords.items():
        is_shelf = location.startswith("shelf")
        is_station = location.startswith("station")
        fill = SHELF_FILL if is_shelf else SURFACE if is_station else AISLE_FILL
        edge = SECONDARY if is_station else HAIRLINE
        ax.add_patch(Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, facecolor=fill, edgecolor=edge, linewidth=1.2))
        label = _short(location) if is_shelf or is_station else location
        ax.text(x - 0.4, y - 0.4, label, fontsize=7.5, color=SECONDARY if is_shelf or is_station else MUTED, ha="left", va="top")

    def item(x: float, y: float, label: str, color: str = ITEM_FILL) -> None:
        ax.add_patch(
            FancyBboxPatch((x - 0.13, y - 0.13), 0.26, 0.26, boxstyle="round,pad=0,rounding_size=0.04",
                           facecolor=color, edgecolor=SURFACE, linewidth=1.5, zorder=6)
        )
        ax.text(x, y, label, fontsize=7, color="#ffffff", ha="center", va="center", zorder=7)

    occupancy: dict[str, int] = {}

    def slot(location: str) -> tuple[float, float]:
        x, y = instance.coords[location]
        index = occupancy.get(location, 0)
        occupancy[location] = index + 1
        return x - 0.2 + 0.28 * (index % 3), y + 0.22

    for item_name, shelf in state.get("stored", []):
        x, y = slot(shelf)
        item(x, y, _short(item_name))
    for item_name, station in state.get("delivered", []):
        x, y = slot(station)
        item(x, y, _short(item_name), GOOD)

    carried = {robot: held for robot, held in state.get("holding", [])}
    for robot, location in state.get("at", []):
        x, y = instance.coords[location]
        color = ROBOT_COLORS[int(robot.removeprefix("robot")) - 1]
        ax.add_patch(Circle((x, y), 0.3, facecolor=color, edgecolor=SURFACE, linewidth=2, zorder=4))
        ax.text(x, y, _short(robot), fontsize=9, color="#ffffff", ha="center", va="center", weight="bold", zorder=5)
        if robot in carried:
            item(x + 0.24, y - 0.24, _short(carried[robot]))

    if failed_action:
        name, *args = failed_action.strip("()").split()
        if name == "move" and len(args) == 3 and args[1] in instance.coords and args[2] in instance.coords:
            start, end = instance.coords[args[1]], instance.coords[args[2]]
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, color=CRITICAL, linewidth=2, zorder=8))
        elif args and args[-1] in instance.coords:
            x, y = instance.coords[args[-1]]
            ax.add_patch(Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, fill=False, edgecolor=CRITICAL, linewidth=2.5, zorder=8))


def _draw_plan(ax, frame: Frame, attempt: Attempt, attempts_total: int, optimal: int) -> None:
    ax.clear()
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0, 1.0, f"Attempt {frame.attempt_number} of {attempts_total}", fontsize=14, color=INK, weight="bold", va="top")
    subtitle = f"{len(attempt.actions)} actions written, optimal is {optimal}"
    ax.text(0, 0.935, subtitle, fontsize=9.5, color=SECONDARY, va="top")

    actions = attempt.actions
    start = max(0, min(frame.current - PLAN_WINDOW // 2, len(actions) - PLAN_WINDOW))
    line_height = 0.038
    top = 0.87
    for offset, action in enumerate(actions[start : start + PLAN_WINDOW]):
        index = start + offset
        failing = frame.status == "failed" and index == frame.current and attempt.failed_step is not None
        if failing:
            color, weight, marker = CRITICAL, "bold", "✗"
        elif index == frame.current and frame.status == "running":
            color, weight, marker = INK, "bold", "›"
        elif index <= frame.current:
            color, weight, marker = SECONDARY, "normal", "✓"
        else:
            color, weight, marker = MUTED, "normal", " "
        ax.text(0, top - offset * line_height, f"{marker} {index + 1:>2}  {action}", fontsize=9.5, color=color,
                family="DejaVu Sans Mono", weight=weight, va="top")

    if frame.status == "failed":
        ax.text(0, 0.285, "✗ Rejected by the validator", fontsize=11, color=CRITICAL, weight="bold", va="top")
        wrapped = "\n".join(textwrap.fill(line, 66) for line in attempt.feedback.splitlines())
        ax.text(0, 0.235, wrapped, fontsize=8.2, color=SECONDARY, va="top", linespacing=1.3)
    elif frame.status == "valid":
        ax.text(0, 0.285, "✓ Valid plan", fontsize=11, color=GOOD_TEXT, weight="bold", va="top")
        ratio = len(actions) / optimal
        ax.text(0, 0.235, f"{len(actions)} actions, {ratio:.2f} times the optimal length.", fontsize=9.5, color=SECONDARY, va="top")


def render_gif(instance: Instance, run: Run, out_path: Path) -> None:
    figure = plt.figure(figsize=(10.2, 5.6), dpi=90, facecolor=SURFACE)
    warehouse_ax = figure.add_axes([0.015, 0.02, 0.5, 0.96])
    plan_ax = figure.add_axes([0.54, 0.03, 0.45, 0.94])

    images, durations = [], []
    for frame in _frames(instance, run):
        attempt = run.attempts[frame.attempt_number - 1]
        failed_action = None
        if frame.status == "failed" and attempt.failed_step is not None:
            failed_action = attempt.actions[attempt.failed_step - 1]
        _draw_warehouse(warehouse_ax, instance, frame.atoms, failed_action)
        _draw_plan(plan_ax, frame, attempt, len(run.attempts), run.optimal_length)
        figure.canvas.draw()
        pixels = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
        images.append(Image.fromarray(pixels.copy()))
        durations.append(frame.duration)
    plt.close(figure)

    # The palette has to come from all frames together: the failure red appears in only a few of them.
    width, height = images[0].size
    montage = Image.new("RGB", (width, height * len(images)))
    for index, image in enumerate(images):
        montage.paste(image, (0, height * index))
    palette = montage.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=1)


def hardest_repaired_run(runs: list[Run]) -> Run | None:
    """The hardest problem the feedback loop fixed, then the shortest story (fewest attempts)."""
    repaired = [r for r in runs if r.arm == FEEDBACK and r.solved and len(r.attempts) > 1]
    return max(repaired, key=lambda r: (r.tier, r.optimal_length, -len(r.attempts)), default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--problem")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--arm", default=FEEDBACK)
    parser.add_argument("--split", default="benchmark")
    args = parser.parse_args()

    runs = load_runs(RESULTS_DIR / "raw" / f"{args.label}.json")
    if args.problem:
        run = next(r for r in runs if (r.problem, r.replicate, r.arm) == (args.problem, args.replicate, args.arm))
    else:
        run = hardest_repaired_run(runs)
        if run is None:
            raise SystemExit("no run was repaired by feedback")
    instance = next(i for i in load_split(args.split) if i.name == run.problem)
    out_path = RESULTS_DIR / "animations" / f"{args.label}_{run.problem}_r{run.replicate}.gif"
    render_gif(instance, run, out_path)
    print(f"{out_path} ({len(run.attempts)} attempts, tier {run.tier})")


if __name__ == "__main__":
    main()
