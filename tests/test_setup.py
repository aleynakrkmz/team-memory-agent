"""Kurulum kontrolü: `uv run pytest` yeşilse ortamın hazır demektir."""

import importlib
import sqlite3
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_python_version_is_3_12():
    assert sys.version_info[:2] == (3, 12), (
        f"Python 3.12 bekleniyordu, {sys.version.split()[0]} bulundu. "
        "Komutları `uv run ...` ile çalıştırdığından emin ol."
    )


@pytest.mark.parametrize("module", ["ollama", "pydantic", "dotenv", "rank_bm25"])
def test_dependencies_importable(module):
    importlib.import_module(module)


def test_sqlite_triggers_supported():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (x INTEGER)")
    con.execute(
        "CREATE TRIGGER no_update BEFORE UPDATE ON t "
        "BEGIN SELECT RAISE(ABORT, 'append-only'); END"
    )
    con.execute("INSERT INTO t VALUES (1)")
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("UPDATE t SET x = 2")


@pytest.mark.parametrize(
    "path",
    [
        "app/ingest", "app/extract", "app/match", "app/route", "app/agent",
        "app/validate", "app/review", "app/commit", "app/store",
        "datagen", "eval", "prompts", "config", "tests", "runs", "data",
        ".env.example",
    ],
)
def test_project_layout(path):
    assert (ROOT / path).exists(), f"{path} eksik"


def test_ollama_reachable():
    """Ollama çalışmıyorsa bu test 'skipped' görünür; kurulum için hata sayılmaz."""
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
    except OSError:
        pytest.skip("Ollama çalışmıyor (henüz kurulmadıysa normal)")
