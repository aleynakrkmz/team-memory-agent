"""Line ids stay aligned with the demo transcript files."""

import json

from app.transcript import parse_speaker_transcript
from tests.demo_support import DEMO_INPUTS, DEMO_KEYS, load_demo


def test_utterance_ids_are_the_file_line_numbers():
    meeting, team = load_demo(1)
    raw_lines = (DEMO_INPUTS / "meeting_01.transcript.txt").read_text(encoding="utf-8").splitlines()
    assert [utterance.id for utterance in meeting.utterances] == [f"L{i}" for i in range(1, len(raw_lines) + 1)]
    assert meeting.utterances[1].speaker == "Zeynep"
    assert meeting.utterances[1].text == raw_lines[1].split(": ", 1)[1]
    assert team.coordinator == "Zeynep"
    assert meeting.meeting_date.isoformat() == "2026-10-29"


def test_ids_match_the_demo_evidence_line_numbers():
    evidence = json.loads((DEMO_KEYS / "meeting_03.evidence.json").read_text(encoding="utf-8"))
    meeting, _team = load_demo(3)
    api_lines = evidence["evidence"]["E9"]
    quoted = " ".join(meeting.utterance(f"L{line}").text for line in api_lines)
    assert "that API thing" in quoted


def test_parser_rejects_a_line_without_a_speaker():
    import pytest

    with pytest.raises(ValueError, match="line 2"):
        parse_speaker_transcript("Zeynep: Hello\nthis line has no speaker\n")
