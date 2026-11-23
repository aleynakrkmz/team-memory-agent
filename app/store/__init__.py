"""Append-only SQLite memory, one database per project. Design §3.4 and §5."""

from app.store.paths import DATA_ROOT, project_db_path, reset_db_path
from app.store.reader import SqliteMemory
from app.store.store import MemoryStore, StoreError

__all__ = [
    "DATA_ROOT",
    "MemoryStore",
    "SqliteMemory",
    "StoreError",
    "project_db_path",
    "reset_db_path",
]
