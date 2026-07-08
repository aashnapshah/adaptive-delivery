"""SQLite data store for the study.

Design goals (this is the "can't lose a response" layer):
  - Single self-contained file (data/study.db), WAL mode → safe concurrent reads + one writer.
  - Every write is transactional and IDEMPOTENT: the client sends a unique client_event_id,
    and inserts are `INSERT OR IGNORE`, so a retry / double-click / refresh never duplicates
    or loses a row.
  - Append-only in spirit: we never UPDATE or DELETE study data.
  - Participant + consent + session bookkeeping so every response is attributable.
  - Plain columns (not blobs) so analysis / CSV export is trivial.

Switch the DB location with the STUDY_DB env var (e.g. a backed-up volume).
"""

from __future__ import annotations

import csv
import datetime
import io
import os
import sqlite3
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("STUDY_DB", os.path.join(HERE, "data", "study.db"))
_WLOCK = threading.Lock()                 # serialize writers (Flask dev server is threaded)

SCHEMA = """
CREATE TABLE IF NOT EXISTS participant (
  pid         TEXT PRIMARY KEY,
  role        TEXT,
  site        TEXT,
  consent_at  TEXT,
  created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session (
  session_id  TEXT PRIMARY KEY,
  pid         TEXT,
  user_agent  TEXT,
  started_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS grade (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  client_event_id TEXT UNIQUE,
  session_id TEXT, pid TEXT, ts TEXT NOT NULL,
  case_id TEXT, design TEXT, test TEXT,
  grade INTEGER, appropriateness TEXT, harm TEXT, latency_ms INTEGER,
  auto_appropriateness REAL, auto_harm REAL, auto_cost REAL, auto_noharm INTEGER
);
CREATE TABLE IF NOT EXISTS sim_event (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  client_event_id TEXT UNIQUE,
  session_id TEXT, pid TEXT, ts TEXT NOT NULL,
  case_id TEXT, test TEXT, choice TEXT, latency_ms INTEGER, shift_min REAL,
  seq INTEGER, fatigue INTEGER,                 -- RL state: sequence index + running interruption count
  appropriateness REAL, harm REAL, cost REAL, auto_noharm INTEGER
);
CREATE TABLE IF NOT EXISTS bench_turn (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  client_event_id TEXT UNIQUE,
  session_id TEXT, pid TEXT, ts TEXT NOT NULL,
  case_id TEXT, action TEXT, query TEXT, latency_ms INTEGER,
  turn INTEGER, total_cost REAL, correct INTEGER, judge_score INTEGER
);
"""

TABLES = ("participant", "session", "grade", "sim_event", "bench_turn")


def _conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


def init():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _WLOCK, _conn() as c:
        c.executescript(SCHEMA)
        for col in ("appropriateness", "harm"):          # migrate existing DBs to the two-axis grade
            try:
                c.execute(f"ALTER TABLE grade ADD COLUMN {col} TEXT")
            except Exception:
                pass


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _insert(table: str, row: dict) -> bool:
    """Idempotent insert. Returns True if a new row was written, False if it was a duplicate
    (same client_event_id) — either way the caller can treat it as success."""
    cols = ", ".join(row)
    qs = ", ".join("?" for _ in row)
    with _WLOCK, _conn() as c:
        cur = c.execute(f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({qs})", tuple(row.values()))
        return cur.rowcount > 0


# ---- participants / sessions ----------------------------------------------

def upsert_participant(pid: str, role: str = None, site: str = None):
    with _WLOCK, _conn() as c:
        c.execute("INSERT OR IGNORE INTO participant (pid, role, site, created_at) VALUES (?,?,?,?)",
                  (pid, role, site, _now()))


def set_consent(pid: str):
    with _WLOCK, _conn() as c:
        c.execute("UPDATE participant SET consent_at=? WHERE pid=? AND consent_at IS NULL", (_now(), pid))


def has_consent(pid: str) -> bool:
    with _conn() as c:
        r = c.execute("SELECT consent_at FROM participant WHERE pid=?", (pid,)).fetchone()
        return bool(r and r["consent_at"])


def ensure_session(session_id: str, pid: str = None, user_agent: str = None):
    with _WLOCK, _conn() as c:
        c.execute("INSERT OR IGNORE INTO session (session_id, pid, user_agent, started_at) VALUES (?,?,?,?)",
                  (session_id, pid, (user_agent or "")[:300], _now()))


# ---- study events (idempotent) --------------------------------------------

def log_grade(**kw) -> bool:
    kw.setdefault("ts", _now())
    return _insert("grade", kw)


def log_sim(**kw) -> bool:
    kw.setdefault("ts", _now())
    return _insert("sim_event", kw)


def log_bench(**kw) -> bool:
    kw.setdefault("ts", _now())
    return _insert("bench_turn", kw)


# ---- export / verification -------------------------------------------------

def counts() -> dict:
    with _conn() as c:
        return {t: c.execute(f"SELECT COUNT(*) n FROM {t}").fetchone()["n"] for t in TABLES}


def export_csv(table: str) -> str:
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}")
    with _conn() as c:
        rows = c.execute(f"SELECT * FROM {table}").fetchall()
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))
    return buf.getvalue()
