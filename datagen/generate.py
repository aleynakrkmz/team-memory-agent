"""Generates the synthetic development or test projects (design §12.1, §12.2, Figure 2).

    # development set: generator A writes the meetings, a different model checks them
    uv run python datagen/generate.py --split dev --projects 7 --generator mistral-nemo:12b --checker llama3.1:8b

    # test set: generator B, test names and tasks; only the WS3 lead reads it
    uv run python datagen/generate.py --split test --projects 3 --generator gemma3:12b --checker llama3.1:8b

    # without a model (template meetings), to try the code
    uv run python datagen/generate.py --split dev --projects 2 --backend mock

Steps for every project (Figure 2):

- **A. Team and scenario** (code, fixed seed): `common.generate_project`. Development and test projects use
  separate name and task lists. The scenario is the answer key: every task change of every meeting.
- **B. Meetings** (the generator model): `common.render` gives the model the participants and the agenda and asks
  for a natural, messy meeting with a label such as [E3] on every line that states an agenda item. Labels are
  removed from the transcript the system reads and kept in the answer key.
- **C. Automatic check**: `common.auto_check`: (i) every event has a labelled line, (ii) owners are named and dates
  say the right day, (iii) an owner left open on purpose is really not named, no label left in the speech,
  (iv) the checker model lists the changes it finds and any change the scenario does not have is flagged.
  A meeting that fails is written again, at most 3 times. If a meeting still fails, the whole project is dropped
  (later meetings depend on it) and the next seed is used. Every attempt and every reason is logged.
- **D. Manual check**: not automatic. `attempts/` keeps every accepted meeting with its labels for reading.
- **E. Freeze**: `manifest.json` lists the SHA-256 of every file, the models with their Ollama digests, the seeds
  and the code commit; `manifest.sha256` is one hash over all of it. Tag the commit in git when the set is frozen.

Output: `datagen/out/<split>/<project>/agent_inputs/...` and `answer_keys/...` (git-ignored), ready for
`eval/harness.py datagen/out/dev/dev01`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

SEED_BASE = {"dev": 1000, "test": 9000}   # dev01 uses seed 1001, test01 uses 9001, ...
MAX_TRIES = 3


def model_digest(client, tag: str | None) -> str | None:
    if client is None or tag is None:
        return None
    try:
        for m in client.list()["models"]:
            if m["model"] == tag:
                return m["digest"]
    except Exception as exc:  # noqa: BLE001
        return f"not recorded ({type(exc).__name__})"
    return "not found"


def generate_one(pid: str, seed: int, split: str, meetings: int, backend: str, client, generator, checker_model,
                 attempts_dir: Path, log: list[dict], say=print) -> tuple[dict, dict, dict] | None:
    """One project. Returns (team, scenario, meetings) or None if a meeting failed every try."""
    team, scenario = C.generate_project(pid, seed, split, n_meetings=meetings)
    kept = {}
    for m in range(1, meetings + 1):
        for attempt in range(1, MAX_TRIES + 1):
            gen_seed = seed * 100 + m * 10 + attempt
            t0 = time.monotonic()
            lines = C.render(team, scenario, m, gen_seed, backend=backend, client=client, model=generator)
            secs = time.monotonic() - t0
            checker = C.make_checker(client, checker_model, gen_seed) if (backend == "ollama" and checker_model) else None
            problems = C.auto_check(team, scenario, m, lines, checker) if lines else [("empty_output", None, "")]
            (attempts_dir / f"m{m:02d}_try{attempt}.txt").write_text(
                "\n".join(f"{s}: {t}" for s, t in lines) + "\n\n# problems\n" +
                "\n".join(f"{r} {e or ''} {d}" for r, e, d in problems), encoding="utf-8")
            log.append({"project": pid, "seed": seed, "meeting": m, "attempt": attempt, "passed": not problems,
                        "reasons": ";".join(sorted({p[0] for p in problems})),
                        "details": " | ".join(f"{r} {e or ''} {d}".strip() for r, e, d in problems)[:500],
                        "lines": len(lines), "seconds": round(secs, 1)})
            reasons = ", ".join(sorted({p[0] for p in problems}))
            say(f"  {pid} meeting {m} try {attempt}: " + (f"failed ({reasons})" if problems else "ok") + f", {secs:.0f} s")
            if not problems:
                kept[m] = lines
                break
        if m not in kept:
            say(f"  {pid}: meeting {m} failed {MAX_TRIES} times, project dropped, next seed")
            return None
    return team, scenario, kept


def write_manifest(out: Path, info: dict) -> str:
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("manifest.json", "manifest.sha256")
                   and "attempts" not in p.parts)
    info["files"] = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    text = json.dumps(info, indent=2, ensure_ascii=False)
    (out / "manifest.json").write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()
    (out / "manifest.sha256").write_text(digest + "  manifest.json\n", encoding="utf-8")
    return digest


def rebuild(src: Path, out: Path) -> int:
    """Writes a generated set again from its saved attempts, with the current label and text cleaning.
    No model is called: the accepted meetings are read from attempts/. Use it after a parser fix.
    The manifest keeps the original models and seeds and records where the set was rebuilt from."""
    old = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    with (src / "generation_log.csv").open(encoding="utf-8") as f:
        log = list(csv.DictReader(f))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "attempts", out / "attempts", dirs_exist_ok=True)
    shutil.copy(src / "generation_log.csv", out / "generation_log.csv")
    for p in old["projects"]:
        pid, seed = p["project"], p["seed"]
        team, scenario = C.generate_project(pid, seed, old["split"], n_meetings=old["meetings_per_project"])
        meetings = {}
        for m in range(1, old["meetings_per_project"] + 1):
            ok = [r for r in log if r["project"] == pid and int(r["seed"]) == seed and int(r["meeting"]) == m
                  and r["passed"] == "True"]
            text = (src / "attempts" / f"{pid}_seed{seed}" / f"m{m:02d}_try{ok[0]['attempt']}.txt").read_text(encoding="utf-8")
            meetings[m] = C.parse_transcript(text.split("\n\n# problems")[0], team)
            problems = C.auto_check(team, scenario, m, meetings[m])
            print(f"  {pid} meeting {m}: " + ("ok" if not problems else f"now fails {problems}"))
        scenario["generator"] = {"backend": old["backend"], "model": old["generator"]["model"],
                                 "checker": old["checker"]["model"]}
        C.write_project(out / pid, pid, team, scenario, meetings)
    info = {k: v for k, v in old.items() if k != "files"}
    info.update(rebuilt_from=old.get("manifest_sha256") or hashlib.sha256((src / "manifest.json").read_bytes()).hexdigest(),
                rebuilt_with_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                                   text=True).stdout.strip(),
                rebuilt_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    digest = write_manifest(out, info)
    print(f"manifest {digest[:16]} -> {out / 'manifest.json'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic projects with their answer keys.")
    ap.add_argument("--rebuild-from", type=Path, help="rewrite a generated set from its attempts/ (no model)")
    ap.add_argument("--split", choices=("dev", "test"))
    ap.add_argument("--projects", type=int, help="how many projects to keep")
    ap.add_argument("--meetings", type=int, default=5)
    ap.add_argument("--backend", choices=("ollama", "mock"), default="ollama")
    ap.add_argument("--generator", help="generator model (dev: mistral-nemo:12b, test: gemma3:12b)")
    ap.add_argument("--checker", help="checker model for check (iv), e.g. llama3.1:8b")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--out", type=Path, help="default datagen/out/<split>")
    ap.add_argument("--max-seeds", type=int, help="give up after this many seeds (default 3 x projects)")
    a = ap.parse_args(argv)
    if a.rebuild_from:
        if not a.out or (a.out.exists() and any(a.out.iterdir())):
            ap.error("--rebuild-from needs a new, empty --out folder")
        return rebuild(a.rebuild_from, a.out)
    if not a.split or a.projects is None:
        ap.error("--split and --projects are required")
    if a.backend == "ollama" and not (a.generator and a.checker):
        ap.error("--generator and --checker are required with the ollama backend")
    if a.split == "test":
        print("Test set: only the WS3 lead reads these meetings. Do not use them to change prompts, rules or limits.")
    out = a.out or ROOT / "datagen" / "out" / a.split
    if out.exists() and any(out.iterdir()):
        ap.error(f"{out} is not empty; a frozen set is never written over. Use --out for a new folder.")
    out.mkdir(parents=True, exist_ok=True)
    client = C.make_client(a.host) if a.backend == "ollama" else None

    log, kept, dropped, started = [], [], [], datetime.now(timezone.utc)
    seed, n, limit = SEED_BASE[a.split], 0, a.max_seeds or 3 * a.projects
    while len(kept) < a.projects and n < limit:
        n += 1
        seed += 1
        pid = f"{a.split}{len(kept) + 1:02d}"
        attempts_dir = out / "attempts" / f"{pid}_seed{seed}"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        print(f"{pid} (seed {seed})")
        result = generate_one(pid, seed, a.split, a.meetings, a.backend, client, a.generator, a.checker,
                              attempts_dir, log)
        if result is None:
            dropped.append(seed)
            continue
        team, scenario, meetings = result
        scenario["generator"] = {"backend": a.backend, "model": a.generator, "checker": a.checker}
        C.write_project(out / pid, pid, team, scenario, meetings)
        kept.append({"project": pid, "seed": seed})

    with (out / "generation_log.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]) if log else ["project"])
        w.writeheader(); w.writerows(log)
    meetings_tried = {(r["project"], r["seed"], r["meeting"]) for r in log}
    first_try = sum(1 for r in log if r["attempt"] == 1 and r["passed"])
    info = {
        "split": a.split, "backend": a.backend,
        "generator": {"model": a.generator, "digest": model_digest(client, a.generator)},
        "checker": {"model": a.checker, "digest": model_digest(client, a.checker)},
        "projects": kept, "dropped_seeds": dropped, "meetings_per_project": a.meetings, "max_tries": MAX_TRIES,
        "first_try_pass_rate": round(first_try / len(meetings_tried), 3) if meetings_tried else None,
        "attempts": len(log),
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    digest = write_manifest(out, info)
    print(f"\nkept {len(kept)} projects, dropped {len(dropped)} seeds; first-try pass rate {info['first_try_pass_rate']}")
    print(f"manifest {digest[:16]} -> {out / 'manifest.json'}")
    return 0 if len(kept) == a.projects else 1


if __name__ == "__main__":
    raise SystemExit(main())
