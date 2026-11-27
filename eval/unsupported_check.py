"""Manual check of committed proposals: is each change really supported by the lines it cites? (design §12)

The harness reports an automatic proxy for the unsupported-update rate. The design asks for a person to check a
sample of 30 committed proposals. This script draws the sample (fixed seed) and writes one HTML page; the reviewer
marks each proposal and downloads a CSV. `--score` reads the CSV back and prints the rate with its denominator.

    uv run python eval/unsupported_check.py runs/eval/*-agent-carry-* --data datagen/out/dev --out check.html
    uv run python eval/unsupported_check.py --score unsupported_check.csv

The page shows the proposal, the lines it cites and the lines around them, never the answer key: the question is
whether the transcript supports the change, not whether the change is in the key.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from pathlib import Path


def committed_proposals(run: Path) -> list[int]:
    """Indexes of the proposals of each meeting that were committed: from reviews/ (newer runs) or rows.csv."""
    out = {}
    if (run / "reviews").is_dir():
        for f in sorted((run / "reviews").glob("m*.json")):
            m = int(f.stem[1:])
            out[m] = [i for i, r in enumerate(json.loads(f.read_text(encoding="utf-8"))) if r.get("committed_as")]
        return out
    with (run / "rows.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("committed") == "True" and r.get("proposal_index") not in ("", None):
                out.setdefault(int(r["meeting"]), []).append(int(r["proposal_index"]))
    return out


def sample(runs: list[Path], data: Path | None, n: int, seed: int) -> list[dict]:
    items = []
    for run in sorted(runs):
        rec = json.loads((run / "run_record.json").read_text(encoding="utf-8"))
        pid = rec["project"]
        base = (data / pid) if data else Path(rec.get("project_path", ""))
        for m, idxs in committed_proposals(run).items():
            proposals = json.loads((run / "previews" / f"m{m}.json").read_text(encoding="utf-8"))["proposals"]
            lines = (base / "agent_inputs" / pid / f"meeting_{m:02d}.transcript.txt").read_text(encoding="utf-8").splitlines()
            for i in sorted(set(idxs)):
                p = proposals[i]
                cited = [int(r["id"][1:]) for r in p.get("evidence_refs", []) if str(r.get("id", "")).startswith("L")]
                items.append({"key": f"{pid}-m{m}-p{i}", "run": run.name, "project": pid, "meeting": m,
                              "configuration": rec["configuration"], "action": p["action"],
                              "task": p.get("task_description") or (p.get("fields") or {}).get("description"),
                              "task_id": p.get("task_id"), "fields": p.get("fields") or {}, "cited": cited,
                              "lines": lines})
    random.Random(seed).shuffle(items)
    return sorted(items[:n], key=lambda x: (x["project"], x["meeting"], x["key"]))


def page(items: list[dict]) -> str:
    cards = []
    for it in items:
        f = it["fields"]
        change = f"{it['action']} <i>{html.escape(str(it['task']))}</i>" + (f" ({it['task_id']})" if it["task_id"] else " (new)")
        if f.get("owner"):
            change += f", owner <b>{html.escape(f['owner'])}</b>"
        if f.get("due_date"):
            change += f", due <b>{f['due_date']}</b>"
        show = sorted({k for c in it["cited"] for k in range(c - 3, c + 4) if 1 <= k <= len(it["lines"])})
        rows = "".join(f'<tr class="{"cite" if k in it["cited"] else ""}"><td class="n">L{k}</td><td>{html.escape(it["lines"][k - 1])}</td></tr>'
                       for k in show)
        cards.append(f'''<div class="card" data-k="{it['key']}"><div class="head"><b>{it['project']} meeting {it['meeting']}</b>
<span class="change">{change}</span></div><table>{rows}</table>
<div class="v"><button data-v="supported">Supported</button><button data-v="partly">Partly (a field is not said)</button>
<button data-v="unsupported">Not supported</button></div><textarea placeholder="Note (optional)"></textarea></div>''')
    data = json.dumps([{k: it[k] for k in ("key", "run", "project", "meeting", "action", "task", "task_id")} for it in items])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unsupported Check</title><style>
:root{{--bg:#f7f6f3;--card:#fff;--ink:#1d1d1b;--mute:#6b6a66;--line:#e4e2dc;--hi:#e8eff6;--acc:#2f5d8a}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#161615;--card:#1f1f1d;--ink:#ecebe7;--mute:#a19f99;--line:#33322f;--hi:#22303d;--acc:#8db6de}}}}
:root[data-theme="dark"]{{--bg:#161615;--card:#1f1f1d;--ink:#ecebe7;--mute:#a19f99;--line:#33322f;--hi:#22303d;--acc:#8db6de}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}main{{max-width:900px;margin:0 auto;padding:20px 16px 60px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:12px 0}}.head{{display:flex;gap:10px;flex-wrap:wrap}}
.change{{color:var(--mute)}}table{{width:100%;border-collapse:collapse;margin-top:8px;font-size:14px}}td{{padding:2px 6px;border-top:1px solid var(--line);vertical-align:top}}
td.n{{color:var(--mute);width:36px}}tr.cite td{{background:var(--hi)}}.v{{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}}
button{{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:5px 9px;cursor:pointer}}
button.on{{background:var(--acc);color:var(--bg);border-color:var(--acc)}}textarea{{width:100%;box-sizing:border-box;margin-top:8px;font:inherit;font-size:14px;
border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink)}}.bar{{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center}}
</style></head><body><main><h1>Unsupported-update check</h1>
<p>{len(items)} committed proposals, drawn at random. Highlighted lines are the ones the proposal cites. For each: does what is
highlighted support the change, including the owner and date? Judge only from the transcript.</p>
<div class="bar"><button id="dl">Download CSV</button><span id="count"></span></div>{''.join(cards)}</main>
<script>
const ITEMS={data};const st={{}};
function upd(){{const v=Object.values(st).map(x=>x.v).filter(Boolean);const u=v.filter(x=>x==="unsupported").length,p=v.filter(x=>x==="partly").length;
document.getElementById("count").textContent=`${{v.length}}/${{ITEMS.length}} checked, ${{u}} not supported, ${{p}} partly`}}
document.querySelectorAll(".card").forEach(c=>{{const k=c.dataset.k;st[k]={{}};
c.querySelectorAll("button").forEach(b=>b.onclick=()=>{{st[k].v=b.dataset.v;c.querySelectorAll("button").forEach(x=>x.classList.toggle("on",x===b));upd()}});
c.querySelector("textarea").onchange=e=>st[k].n=e.target.value}});
document.getElementById("dl").onclick=()=>{{const rows=[["key","run","project","meeting","action","task","task_id","verdict","note"]].concat(
ITEMS.map(i=>[i.key,i.run,i.project,i.meeting,i.action,i.task,i.task_id||"",(st[i.key]||{{}}).v||"",(st[i.key]||{{}}).n||""]));
const csv=rows.map(r=>r.map(c=>'"'+String(c).replace(/"/g,'""')+'"').join(",")).join("\\n");
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{{type:"text/csv"}}));a.download="unsupported_check.csv";a.click()}};upd();
</script></body></html>'''


def score(path: Path) -> str:
    with path.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["verdict"]]
    n = len(rows)
    un = sum(r["verdict"] == "unsupported" for r in rows)
    pa = sum(r["verdict"] == "partly" for r in rows)
    if not n:
        return "no verdicts in the file"
    return (f"checked {n}: not supported {un} ({un / n:.0%}), partly supported {pa} ({pa / n:.0%}), "
            f"supported {n - un - pa} ({(n - un - pa) / n:.0%})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sample committed proposals for a manual support check, or score the check.")
    ap.add_argument("runs", type=Path, nargs="*", help="run folders made by eval/harness.py")
    ap.add_argument("--data", type=Path, help="folder that holds the projects, if the runs were made on another machine")
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=773)
    ap.add_argument("--out", type=Path, default=Path("unsupported_check.html"))
    ap.add_argument("--score", type=Path, help="a CSV downloaded from the page: print the rates")
    a = ap.parse_args(argv)
    if a.score:
        print(score(a.score))
        return 0
    items = sample(a.runs, a.data, a.n, a.seed)
    a.out.write_text(page(items), encoding="utf-8")
    print(f"{len(items)} proposals -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
