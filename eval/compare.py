"""One table that compares evaluation runs (for the Week 10 write-up).

    uv run python eval/compare.py runs/eval/*          # every run folder given
    uv run python eval/compare.py runs/eval/* --out docs/first_evaluation_table.md

Every row says the configuration, the mode, the execution mode and the model, so that numbers from different
modes are never mixed up (a replay or mock number is not a live number).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(folders: list[Path]) -> list[dict]:
    out = []
    for f in folders:
        s, r = f / "scores.json", f / "run_record.json"
        if not (s.exists() and r.exists()):
            continue
        scores, record = json.loads(s.read_text()), json.loads(r.read_text())
        a = scores["change_f1"]["all"]
        out.append({
            "project": record["project"], "configuration": record["configuration"], "mode": record["mode"],
            "execution": record["execution_mode"], "model": record["model"].get("model") or "none",
            "precision": a["precision"], "recall": a["recall"], "f1": a["f1"],
            "tp/fp/fn": f"{a['tp']}/{a['fp']}/{a['fn']}",
            "unsupported (proxy)": scores["change_f1"]["unsupported_rate_proxy"],
            "task state": (scores.get("task_state") or {}).get("mean_accuracy", ""),
            "stop reasons": ", ".join(sorted({m["stop_reason"] for m in scores["meetings"]})),
            "fallbacks": sum(m["extraction_fallback"] for m in scores["meetings"]),
            "seconds": round(sum(m["seconds"] for m in scores["meetings"]), 1),
            "run": f.name,
        })
    return sorted(out, key=lambda x: (x["project"], x["mode"], x["configuration"], x["run"]))


def totals(table: list[dict]) -> list[dict]:
    """Micro F1 over all projects, one row per configuration, mode, execution mode and model.
    The counts are added first and F1 is computed once, so a large project weighs more than a small one."""
    groups: dict[tuple, list[int]] = {}
    for r in table:
        key = (r["configuration"], r["mode"], r["execution"], r["model"])
        tp, fp, fn = (int(x) for x in r["tp/fp/fn"].split("/"))
        g = groups.setdefault(key, [0, 0, 0, 0])
        g[0] += tp; g[1] += fp; g[2] += fn; g[3] += 1
    out = []
    for (config, mode, execution, model), (tp, fp, fn, n) in sorted(groups.items()):
        p = round(tp / (tp + fp), 3) if tp + fp else None
        rc = round(tp / (tp + fn), 3) if tp + fn else None
        f1 = round(2 * tp / (2 * tp + fp + fn), 3) if tp else 0.0
        out.append({"configuration": config, "mode": mode, "execution": execution, "model": model, "projects": n,
                    "precision": p, "recall": rc, "f1": f1, "tp/fp/fn": f"{tp}/{fp}/{fn}"})
    return out


def markdown(table: list[dict]) -> str:
    if not table:
        return "No runs found.\n"
    cols = list(table[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in table]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare evaluation runs in one table.")
    ap.add_argument("runs", type=Path, nargs="+", help="run folders made by eval/harness.py")
    ap.add_argument("--out", type=Path, help="also write the table to this Markdown file")
    a = ap.parse_args(argv)
    table = rows(a.runs)
    text = markdown(table)
    if len({r["project"] for r in table}) > 1:
        text = "All projects (micro F1):\n\n" + markdown(totals(table)) + "\nPer project:\n\n" + text
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
