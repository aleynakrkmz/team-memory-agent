"""Tests for the manual unsupported-update check (eval/unsupported_check.py)."""

from eval.harness import run_trial
from eval.unsupported_check import main, page, sample, score
from tests.demo_support import ROOT

DEMO = ROOT / "samples" / "demo"


def test_sample_page_and_score(tmp_path):
    run = run_trial(DEMO, "b1", "carry", out_root=tmp_path, say=lambda _: None)
    items = sample([run], None, n=3, seed=1)
    assert 1 <= len(items) <= 3 and all(it["cited"] for it in items)
    assert "data-k" in page(items)
    csv_path = tmp_path / "check.csv"
    csv_path.write_text('key,run,project,meeting,action,task,task_id,verdict,note\n'
                        '"a","r","demo","1","create","x","","supported",""\n'
                        '"b","r","demo","1","create","y","","unsupported",""\n'
                        '"c","r","demo","1","create","z","","",""\n', encoding="utf-8")
    assert score(csv_path).startswith("checked 2: not supported 1 (50%)")
    assert main([str(run), "-n", "2", "--out", str(tmp_path / "p.html")]) == 0
