"""Tests for datagen/generate.py with the template backend (no model needed)."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datagen"))
import common as C  # noqa: E402
import generate as G  # noqa: E402


def run(out, *extra):
    return G.main(["--split", "dev", "--projects", "2", "--backend", "mock", "--out", str(out), *extra])


def test_writes_projects_and_manifest(tmp_path):
    assert run(tmp_path / "dev") == 0
    out = tmp_path / "dev"
    for pid in ("dev01", "dev02"):
        assert (out / pid / "agent_inputs" / pid / "meeting_05.transcript.txt").exists()
        assert (out / pid / "answer_keys" / pid / "scenario.json").exists()
    info = json.loads((out / "manifest.json").read_text())
    assert [p["seed"] for p in info["projects"]] == [1001, 1002]
    for rel, digest in info["files"].items():
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == digest
    assert not any("attempts" in rel for rel in info["files"])
    assert (out / "manifest.sha256").read_text().split()[0] == \
        hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()
    # no labels in what the system reads
    text = (out / "dev01" / "agent_inputs" / "dev01" / "meeting_03.transcript.txt").read_text()
    assert "[E" not in text


def test_same_seed_same_data(tmp_path):
    run(tmp_path / "a"); run(tmp_path / "b")
    a = json.loads((tmp_path / "a" / "manifest.json").read_text())["files"]
    b = json.loads((tmp_path / "b" / "manifest.json").read_text())["files"]
    assert a == b


def test_never_writes_over_a_set(tmp_path):
    run(tmp_path / "dev")
    with pytest.raises(SystemExit):
        run(tmp_path / "dev")


def test_retry_then_drop(tmp_path, monkeypatch):
    real = C.render
    calls = []

    def flaky(team, scenario, m, seed, **kw):
        calls.append((seed // 100, m, seed % 10))
        if seed // 100 == 1001 and m == 2:     # project seed 1001, meeting 2 fails every try
            return []
        if seed // 100 == 1002 and m == 1 and seed % 10 == 1:  # seed 1002 meeting 1 fails once
            return []
        return real(team, scenario, m, seed, **kw)

    monkeypatch.setattr(C, "render", flaky)
    assert run(tmp_path / "dev") == 0
    out = tmp_path / "dev"
    info = json.loads((out / "manifest.json").read_text())
    assert info["dropped_seeds"] == [1001]
    assert [p["seed"] for p in info["projects"]] == [1002, 1003]
    assert [c for c in calls if c[0] == 1001 and c[1] == 2] == [(1001, 2, 1), (1001, 2, 2), (1001, 2, 3)]
    log = (out / "generation_log.csv").read_text().splitlines()
    assert any(r.startswith("dev01,1002,1,1,False,empty_output") for r in log)
    assert any(r.startswith("dev01,1002,1,2,True") for r in log)
    assert (out / "attempts" / "dev01_seed1001" / "m02_try3.txt").exists()


def test_gives_up_after_max_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "render", lambda *a, **k: [])
    assert run(tmp_path / "dev", "--max-seeds", "2") == 1
    assert json.loads((tmp_path / "dev" / "manifest.json").read_text())["projects"] == []


def test_ollama_needs_models(tmp_path):
    with pytest.raises(SystemExit):
        G.main(["--split", "dev", "--projects", "1", "--out", str(tmp_path / "x")])


class FakeOllama:
    """Answers like Ollama: the generator writes the template meeting as text, the checker finds nothing new."""

    def __init__(self):
        self.models = []

    def chat(self, model, messages, options):
        self.models.append(model)
        if model == "checker":
            return {"message": {"content": "Here you go:\n[]"}}
        team, scenario, m = self.current
        lines = C.render_mock(team, scenario, m, __import__("random").Random(options["seed"]))
        return {"message": {"content": "**Transcript**\n" + "\n".join(f"**{s}:** {t}" for s, t in lines)}}

    def list(self):
        return {"models": [{"model": "gen", "digest": "abc123"}, {"model": "checker", "digest": "def456"}]}


def test_ollama_path_with_a_fake_client(tmp_path, monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(C, "make_client", lambda host: fake)
    real = C.render

    def remember(team, scenario, m, seed, **kw):
        fake.current = (team, scenario, m)
        return real(team, scenario, m, seed, **kw)

    monkeypatch.setattr(C, "render", remember)
    assert G.main(["--split", "dev", "--projects", "1", "--generator", "gen", "--checker", "checker",
                   "--out", str(tmp_path / "dev")]) == 0
    info = json.loads((tmp_path / "dev" / "manifest.json").read_text())
    assert info["generator"] == {"model": "gen", "digest": "abc123"}
    assert info["checker"] == {"model": "checker", "digest": "def456"}
    assert fake.models.count("gen") == 5 and fake.models.count("checker") == 5
    text = (tmp_path / "dev" / "dev01" / "agent_inputs" / "dev01" / "meeting_01.transcript.txt").read_text()
    assert "*" not in text and "[E" not in text


def test_labels_in_every_form_are_removed():
    cases = ["By Thursday. [E1]", "(E4) Move it to Can.", "E5: Ayşe, take it?", "Done [E4][E5]", "Fine **[E6]**",
             "Agreed - E8", "Ok [Event E3]", "Okay. [e4]", "E2E tests are flaky."]
    transcript, evidence = C.split_tags([("Ali", c) for c in cases])
    assert [t for _, t in transcript] == ["By Thursday.", "Move it to Can.", "Ayşe, take it?", "Done", "Fine",
                                          "Agreed", "Ok", "Okay.", "E2E tests are flaky."]
    # a label at the start of a line belongs to the line before: "(E4) Move it" -> line 1, "E5: Ayşe" -> line 2
    assert evidence["E4"] == [1, 4, 8] and evidence["E5"] == [2, 4]


def test_owner_on_the_line_before_the_label():
    team, scenario = C.generate_project("dev01", 1001, "dev", n_meetings=5)
    e1 = next(e for e in scenario["events"] if e["event_id"] == "E1")
    assert (e1["owner"], e1["due_phrase"]) == ("Ayşe", "by Thursday")
    lines = [("Ali", "First, the search filters. Who can take them?"), ("Zeynep", "I'm full this week."),
             ("Ayşe", "Alright, I'll take it on. When do you need it by?"), ("Ali", "By Thursday, please. [E1]")]
    transcript, evidence = C.split_tags(lines)
    assert C.complete_evidence(team, scenario, 1, transcript, evidence)["E1"] == [3, 4]
    assert not [p for p in C.auto_check(team, scenario, 1, lines) if p[1] == "E1"]


def test_quotes_and_stage_directions_are_cleaned():
    team = {"members": [{"name": "Ali"}, {"name": "Burak"}]}
    raw = 'Ali: "Great. Next, E9. Burak, Saturday?" [E9]\nBurak: *sighs* Fine. [E9]\nAli: "Thanks." '
    transcript, evidence = C.split_tags(C.parse_transcript(raw, team))
    assert transcript == [("Ali", "Great. Next. Burak, Saturday?"), ("Burak", "Fine."), ("Ali", "Thanks.")]
    assert evidence == {"E9": [1, 2]}


def test_label_alone_or_first_belongs_to_the_line_before():
    lines = [("Ece", "Yeah, it's all wrapped up."), ("Burak", "[E4] Great job, Ece. Next up, the schema?"),
             ("Mert", "Done too."), ("Burak", "[E5]"), ("Burak", "Good, that's all.")]
    transcript, evidence = C.split_tags(lines)
    assert [t for _, t in transcript] == ["Yeah, it's all wrapped up.", "Great job, Ece. Next up, the schema?",
                                          "Done too.", "Good, that's all."]
    assert evidence == {"E4": [1], "E5": [3]}


def test_stage_directions_are_removed_but_labels_kept():
    team = {"members": [{"name": "Ali"}]}
    raw = "Ali: Okay, noted. [E13] *(Dropped)* for now.\nAli: Push it to... (checks calendar)...by the 13th. (E15)"
    transcript, evidence = C.split_tags(C.parse_transcript(raw, team))
    assert [t for _, t in transcript] == ["Okay, noted. for now.", "Push it to...by the 13th."]
    assert evidence == {"E13": [1], "E15": [2]}
