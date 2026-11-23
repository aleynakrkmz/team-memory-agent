"""Where each database file lives. Design §9 (file table).

    data/<project_id>/<run_id>/memory.sqlite                     carry-forward
    data/<project_id>/<run_id>/reset/<meeting_id>/memory.sqlite  reset-to-gold

Ids become folder names, so they are limited to letters, digits, `_`, `-`
and `.`, and cannot be `.` or `..`. That keeps every file inside the data
root and keeps projects and runs apart.
"""

import re
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DB_FILENAME = "memory.sqlite"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _check_id(kind: str, value: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.match(value) or value in (".", ".."):
        raise ValueError(f"{kind} {value!r} must use only letters, digits, '_', '-' or '.'")
    return value


def project_db_path(project_id: str, run_id: str, *, data_root: Path = DATA_ROOT) -> Path:
    """The one database for a normal project, or for one carry-forward trial."""
    return Path(data_root) / _check_id("project_id", project_id) / _check_id("run_id", run_id) / DB_FILENAME


def reset_db_path(project_id: str, run_id: str, meeting_id: str, *, data_root: Path = DATA_ROOT) -> Path:
    """A fresh database for one meeting in a reset-to-gold trial."""
    return (
        Path(data_root)
        / _check_id("project_id", project_id)
        / _check_id("run_id", run_id)
        / "reset"
        / _check_id("meeting_id", str(meeting_id))
        / DB_FILENAME
    )
