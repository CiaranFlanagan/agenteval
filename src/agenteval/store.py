"""SQLite-backed storage for traced runs.

SQLite rather than a service: the whole point is that instrumenting an agent
should cost nothing to set up. Runs are queried by (name, version), which is
the access pattern the eval runner needs, so that pair is indexed.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from .models import Run, Span

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    version    TEXT NOT NULL,
    case_id    TEXT,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    inputs     TEXT NOT NULL,
    output     TEXT,
    error      TEXT
);

CREATE TABLE IF NOT EXISTS spans (
    span_id       TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    parent_id     TEXT,
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0,
    error         TEXT,
    attributes    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_runs_name_version ON runs(name, version);
CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id);
"""


class Store:
    """Persistent trace storage."""

    def __init__(self, path: str | Path = "agenteval.db") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def save_run(self, run: Run) -> None:
        """Persist a run and its spans in one transaction."""
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, name, version, case_id, started_at, ended_at, inputs, output, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.run_id,
                    run.name,
                    run.version,
                    run.case_id,
                    run.started_at.isoformat(),
                    run.ended_at.isoformat() if run.ended_at else None,
                    json.dumps(run.inputs),
                    run.output,
                    run.error,
                ),
            )
            self._conn.executemany(
                """INSERT OR REPLACE INTO spans
                   (span_id, run_id, parent_id, name, kind, started_at, ended_at,
                    input_tokens, output_tokens, cost_usd, error, attributes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        s.span_id,
                        s.run_id,
                        s.parent_id,
                        s.name,
                        s.kind,
                        s.started_at.isoformat(),
                        s.ended_at.isoformat() if s.ended_at else None,
                        s.input_tokens,
                        s.output_tokens,
                        s.cost_usd,
                        s.error,
                        json.dumps(s.attributes),
                    )
                    for s in run.spans
                ],
            )

    def runs(self, name: str | None = None, version: str | None = None) -> list[Run]:
        """Load runs, optionally filtered by agent name and version."""
        clauses, params = [], []
        if name is not None:
            clauses.append("name = ?")
            params.append(name)
        if version is not None:
            clauses.append("version = ?")
            params.append(version)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        rows = self._conn.execute(
            f"SELECT * FROM runs{where} ORDER BY started_at", params
        ).fetchall()
        return [self._load_run(r) for r in rows]

    def versions(self, name: str | None = None) -> list[str]:
        """Distinct versions recorded, oldest first."""
        where, params = ("WHERE name = ?", [name]) if name else ("", [])
        rows = self._conn.execute(
            f"SELECT version, MIN(started_at) AS first_seen FROM runs {where} "
            "GROUP BY version ORDER BY first_seen",
            params,
        ).fetchall()
        return [r["version"] for r in rows]

    def _load_run(self, row: sqlite3.Row) -> Run:
        run = Run(
            name=row["name"],
            version=row["version"],
            run_id=row["run_id"],
            case_id=row["case_id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            inputs=json.loads(row["inputs"]),
            output=row["output"],
            error=row["error"],
        )
        span_rows = self._conn.execute(
            "SELECT * FROM spans WHERE run_id = ? ORDER BY started_at", (run.run_id,)
        ).fetchall()
        run.spans = [
            Span(
                name=s["name"],
                kind=s["kind"],
                run_id=s["run_id"],
                span_id=s["span_id"],
                parent_id=s["parent_id"],
                started_at=datetime.fromisoformat(s["started_at"]),
                ended_at=datetime.fromisoformat(s["ended_at"]) if s["ended_at"] else None,
                input_tokens=s["input_tokens"],
                output_tokens=s["output_tokens"],
                cost_usd=s["cost_usd"],
                error=s["error"],
                attributes=json.loads(s["attributes"]),
            )
            for s in span_rows
        ]
        return run
