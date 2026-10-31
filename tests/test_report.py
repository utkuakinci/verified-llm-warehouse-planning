import json

import warehouse_planning.report as report
from warehouse_planning.loop import BLIND, FEEDBACK, Attempt, Run, save_runs
from warehouse_planning.metrics import summarize


def attempt(outcome: str, actions: int) -> Attempt:
    return Attempt("", outcome, ["(x)"] * actions, None, "", [])


def build_from(runs, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path)
    summary = summarize(runs, 1) | {"model": "test", "settings": {"replicates": 1, "temperature": 0.3}}
    (tmp_path / "t_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    save_runs(runs, tmp_path / "raw" / "t.json")
    report.build("t")
    return (tmp_path / "REPORT_t.md").read_text(encoding="utf-8")


def test_report_survives_when_only_the_blind_arm_solves(tmp_path, monkeypatch):
    runs = [
        Run("p1", 1, 0, FEEDBACK, 8, [attempt("goal", 2), attempt("goal", 2)]),
        Run("p1", 1, 0, BLIND, 8, [attempt("goal", 2), attempt("valid", 8)]),
    ]
    text = build_from(runs, tmp_path, monkeypatch)
    assert "Plan quality" not in text
    assert "| **Valid after up to 1 retries** | **0.0%** | **100.0%** |" in text


def test_report_shows_plan_quality_when_feedback_solves(tmp_path, monkeypatch):
    runs = [
        Run("p1", 1, 0, FEEDBACK, 8, [attempt("goal", 2), attempt("valid", 10)]),
        Run("p1", 1, 0, BLIND, 8, [attempt("goal", 2), attempt("goal", 2)]),
    ]
    text = build_from(runs, tmp_path, monkeypatch)
    assert "Plan quality" in text
    assert (tmp_path / "figures" / "t_plan_length.png").exists()
