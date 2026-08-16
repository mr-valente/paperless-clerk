from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ACTIVE_STATUSES = ("queued", "running", "retry_wait")
TERMINAL_STATUSES = ("completed", "failed", "needs_review", "cancelled")


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    document_id INTEGER NOT NULL,
    document_title TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL CHECK(mode IN ('full', 'ocr', 'metadata')),
    status TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'queued',
    progress_current INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0,
    attempt INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    next_run_at REAL NOT NULL,
    lease_until REAL,
    worker_id TEXT,
    error_code TEXT,
    error_message TEXT,
    source_hash TEXT,
    ocr_fingerprint TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_active_document
ON jobs(document_id) WHERE status IN ('queued', 'running', 'retry_wait');
CREATE INDEX IF NOT EXISTS ix_jobs_claim ON jobs(status, next_run_at, created_at);
CREATE INDEX IF NOT EXISTS ix_jobs_document ON jobs(document_id, created_at DESC);

CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    level TEXT NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    data_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_job ON job_events(job_id, id DESC);

CREATE TABLE IF NOT EXISTS page_results (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    text TEXT,
    error TEXT,
    duration_ms INTEGER,
    updated_at REAL NOT NULL,
    PRIMARY KEY(job_id, page_number)
);

CREATE TABLE IF NOT EXISTS conflicts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL,
    document_title TEXT NOT NULL DEFAULT '',
    existing_text TEXT NOT NULL,
    generated_text TEXT NOT NULL,
    score REAL NOT NULL,
    metrics_json TEXT NOT NULL,
    diff_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    resolution TEXT,
    conflict_tag_id INTEGER,
    created_at REAL NOT NULL,
    resolved_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_conflict_open_document
ON conflicts(document_id) WHERE status = 'open';
CREATE INDEX IF NOT EXISTS ix_conflicts_status ON conflicts(status, created_at DESC);

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL,
    document_title TEXT NOT NULL DEFAULT '',
    proposal_json TEXT NOT NULL,
    applied_json TEXT NOT NULL,
    rationale_json TEXT NOT NULL,
    before_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_decisions_created ON decisions(created_at DESC);
CREATE INDEX IF NOT EXISTS ix_decisions_document ON decisions(document_id, created_at DESC);

CREATE TABLE IF NOT EXISTS document_state (
    document_id INTEGER PRIMARY KEY,
    paperless_modified TEXT,
    source_hash TEXT,
    last_job_id TEXT,
    last_status TEXT NOT NULL,
    processed_at REAL NOT NULL
);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        # WAL + NORMAL preserves crash-safe committed transactions while
        # avoiding a full filesystem sync for every completed OCR page.
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        return connection

    def initialize(self) -> None:
        with self._init_lock, self.connect() as connection:
            connection.executescript(SCHEMA)
            job_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "ocr_fingerprint" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN ocr_fingerprint TEXT")
            now = time.time()
            connection.execute(
                "UPDATE jobs SET status='queued', phase='recovered', worker_id=NULL, "
                "lease_until=NULL, next_run_at=?, updated_at=? WHERE status='running'",
                (now, now),
            )
            connection.execute(
                "UPDATE conflicts SET resolution=NULL WHERE status='open' "
                "AND resolution LIKE 'pending:%'"
            )

    def get_setting(self, key: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        now = time.time()
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO settings(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, now),
            )

    def enqueue_job(
        self,
        document_id: int,
        mode: str,
        max_attempts: int,
        *,
        document_title: str = "",
    ) -> tuple[dict[str, Any], bool]:
        now = time.time()
        job_id = str(uuid.uuid4())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM jobs WHERE document_id=? AND status IN ('queued','running','retry_wait') "
                "ORDER BY created_at DESC LIMIT 1",
                (document_id,),
            ).fetchone()
            if existing:
                connection.commit()
                return dict(existing), False
            connection.execute(
                "INSERT INTO jobs(id,document_id,document_title,mode,status,phase,max_attempts,"
                "next_run_at,created_at,updated_at) VALUES(?,?,?,?,'queued','queued',?,?,?,?)",
                (
                    job_id,
                    document_id,
                    document_title,
                    mode,
                    max_attempts,
                    now,
                    now,
                    now,
                ),
            )
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            connection.commit()
        self.add_event(
            job_id, "info", "enqueued", f"Document {document_id} queued for {mode} processing"
        )
        return dict(row), True

    def claim_job(self, worker_id: str, lease_seconds: int) -> dict[str, Any] | None:
        now = time.time()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE jobs SET status='queued', phase='recovered', worker_id=NULL, lease_until=NULL, "
                "next_run_at=?, updated_at=? WHERE status='running' AND lease_until IS NOT NULL AND lease_until < ?",
                (now, now, now),
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE status IN ('queued','retry_wait') AND next_run_at <= ? "
                "ORDER BY next_run_at, created_at LIMIT 1",
                (now,),
            ).fetchone()
            if not row:
                connection.commit()
                return None
            started_at = row["started_at"] or now
            connection.execute(
                "UPDATE jobs SET status='running', phase='starting', attempt=attempt+1, worker_id=?, "
                "lease_until=?, started_at=?, updated_at=?, error_code=NULL, error_message=NULL WHERE id=?",
                (worker_id, now + lease_seconds, started_at, now, row["id"]),
            )
            claimed = connection.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
            connection.commit()
        return dict(claimed)

    def heartbeat(self, job_id: str, lease_seconds: int) -> None:
        now = time.time()
        with self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET lease_until=?, updated_at=? WHERE id=? AND status='running'",
                (now + lease_seconds, now, job_id),
            )

    def update_job(
        self,
        job_id: str,
        *,
        phase: str | None = None,
        current: int | None = None,
        total: int | None = None,
        title: str | None = None,
        source_hash: str | None = None,
        ocr_fingerprint: str | None = None,
    ) -> None:
        fields: list[str] = ["updated_at=?"]
        values: list[Any] = [time.time()]
        for column, value in (
            ("phase", phase),
            ("progress_current", current),
            ("progress_total", total),
            ("document_title", title),
            ("source_hash", source_hash),
            ("ocr_fingerprint", ocr_fingerprint),
        ):
            if value is not None:
                fields.append(f"{column}=?")
                values.append(value)
        values.append(job_id)
        with self.connect() as connection:
            connection.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id=?", values)

    def finish_job(self, job_id: str, status: str = "completed", phase: str = "complete") -> None:
        now = time.time()
        with self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET status=?, phase=?, lease_until=NULL, worker_id=NULL, "
                "completed_at=?, updated_at=? WHERE id=?",
                (status, phase, now, now, job_id),
            )

    def fail_or_retry(self, job_id: str, code: str, message: str, retryable: bool) -> str:
        now = time.time()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT attempt,max_attempts FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if not row:
                connection.rollback()
                return "missing"
            should_retry = retryable and row["attempt"] < row["max_attempts"]
            if should_retry:
                delay = min(300, 5 * (2 ** max(0, row["attempt"] - 1)))
                status, phase, completed_at = "retry_wait", "retry_wait", None
                next_run = now + delay
            else:
                status, phase, completed_at = "failed", "failed", now
                next_run = now
            connection.execute(
                "UPDATE jobs SET status=?,phase=?,next_run_at=?,lease_until=NULL,worker_id=NULL,"
                "error_code=?,error_message=?,completed_at=?,updated_at=? WHERE id=?",
                (status, phase, next_run, code, message[:2000], completed_at, now, job_id),
            )
            connection.commit()
        self.add_event(
            job_id, "warning" if should_retry else "error", status, message, {"code": code}
        )
        return status

    def mark_needs_review(self, job_id: str, phase: str, message: str) -> None:
        now = time.time()
        with self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET status='needs_review',phase=?,lease_until=NULL,worker_id=NULL,"
                "error_message=?,completed_at=?,updated_at=? WHERE id=?",
                (phase, message[:2000], now, now, job_id),
            )
        self.add_event(job_id, "warning", "needs_review", message)

    def retry_job(self, job_id: str) -> dict[str, Any] | None:
        now = time.time()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if (
                not row
                or row["status"] not in TERMINAL_STATUSES
                or (row["status"] == "needs_review" and row["phase"] == "ocr_conflict")
            ):
                connection.rollback()
                return None
            conflict = connection.execute(
                "SELECT 1 FROM conflicts WHERE document_id=? AND status='open' LIMIT 1",
                (row["document_id"],),
            ).fetchone()
            if conflict:
                connection.rollback()
                return None
            active = connection.execute(
                "SELECT id FROM jobs WHERE document_id=? AND status IN ('queued','running','retry_wait')",
                (row["document_id"],),
            ).fetchone()
            if active:
                connection.rollback()
                return None
            connection.execute(
                "UPDATE jobs SET status='queued',phase='queued',attempt=0,next_run_at=?,lease_until=NULL,"
                "worker_id=NULL,error_code=NULL,error_message=NULL,completed_at=NULL,updated_at=? WHERE id=?",
                (now, now, job_id),
            )
            updated = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            connection.commit()
        self.add_event(job_id, "info", "retried", "Job manually queued for retry")
        return dict(updated)

    def cancel_job(self, job_id: str) -> bool:
        now = time.time()
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status='cancelled',phase='cancelled',completed_at=?,updated_at=? "
                "WHERE id=? AND status IN ('queued','retry_wait')",
                (now, now, job_id),
            )
        if cursor.rowcount:
            self.add_event(job_id, "info", "cancelled", "Job cancelled")
        return bool(cursor.rowcount)

    def get_job(self, job_id: str, *, include_events: bool = False) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            result = dict(row)
            if include_events:
                decision = connection.execute(
                    "SELECT id FROM decisions WHERE job_id=? ORDER BY created_at DESC LIMIT 1",
                    (job_id,),
                ).fetchone()
                result["decision_id"] = str(decision["id"]) if decision else None
                result["events"] = [
                    self._event_row(item)
                    for item in connection.execute(
                        "SELECT * FROM job_events WHERE job_id=? ORDER BY id DESC LIMIT 100",
                        (job_id,),
                    ).fetchall()
                ]
                result["page_failures"] = [
                    dict(item)
                    for item in connection.execute(
                        "SELECT page_number,status,attempts,error,duration_ms FROM page_results "
                        "WHERE job_id=? AND status != 'complete' ORDER BY page_number",
                        (job_id,),
                    ).fetchall()
                ]
        return result

    def list_jobs(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM jobs"
        values: list[Any] = []
        if status:
            sql += " WHERE status=?"
            values.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        values.append(limit)
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(sql, values).fetchall()]

    def add_event(
        self,
        job_id: str,
        level: str,
        event_type: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO job_events(job_id,level,event_type,message,data_json,created_at) VALUES(?,?,?,?,?,?)",
                (job_id, level, event_type, message[:1000], _json(data or {}), time.time()),
            )

    @staticmethod
    def _event_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["data"] = json.loads(item.pop("data_json"))
        return item

    def upsert_page(
        self,
        job_id: str,
        page_number: int,
        status: str,
        attempts: int,
        *,
        text: str | None = None,
        error: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO page_results(job_id,page_number,status,attempts,text,error,duration_ms,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(job_id,page_number) DO UPDATE SET "
                "status=excluded.status,attempts=excluded.attempts,text=excluded.text,error=excluded.error,"
                "duration_ms=excluded.duration_ms,updated_at=excluded.updated_at",
                (job_id, page_number, status, attempts, text, error, duration_ms, time.time()),
            )

    def page_results(self, job_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM page_results WHERE job_id=? ORDER BY page_number", (job_id,)
                ).fetchall()
            ]

    def clear_pages(self, job_id: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM page_results WHERE job_id=?", (job_id,))

    def create_conflict(
        self,
        *,
        job_id: str,
        document_id: int,
        document_title: str,
        existing_text: str,
        generated_text: str,
        score: float,
        metrics: dict[str, Any],
        diff: list[dict[str, str]],
        tag_id: int | None,
    ) -> dict[str, Any]:
        conflict_id = str(uuid.uuid4())
        now = time.time()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            prior = connection.execute(
                "SELECT id FROM conflicts WHERE document_id=? AND status='open'", (document_id,)
            ).fetchone()
            if prior:
                connection.execute(
                    "UPDATE conflicts SET job_id=?,document_title=?,existing_text=?,generated_text=?,score=?,"
                    "metrics_json=?,diff_json=?,conflict_tag_id=?,resolution=NULL,created_at=? WHERE id=?",
                    (
                        job_id,
                        document_title,
                        existing_text,
                        generated_text,
                        score,
                        _json(metrics),
                        _json(diff),
                        tag_id,
                        now,
                        prior["id"],
                    ),
                )
                conflict_id = prior["id"]
            else:
                connection.execute(
                    "INSERT INTO conflicts(id,job_id,document_id,document_title,existing_text,generated_text,"
                    "score,metrics_json,diff_json,conflict_tag_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        conflict_id,
                        job_id,
                        document_id,
                        document_title,
                        existing_text,
                        generated_text,
                        score,
                        _json(metrics),
                        _json(diff),
                        tag_id,
                        now,
                    ),
                )
            row = connection.execute(
                "SELECT * FROM conflicts WHERE id=?", (conflict_id,)
            ).fetchone()
            connection.commit()
        return self._conflict_row(row, detail=True)

    def list_conflicts(self, status: str = "open", limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id,job_id,document_id,document_title,score,metrics_json,status,resolution,"
                "conflict_tag_id,created_at,resolved_at FROM conflicts WHERE status=? "
                "ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        return [self._conflict_row(row, detail=False) for row in rows]

    def get_conflict(self, conflict_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM conflicts WHERE id=?", (conflict_id,)
            ).fetchone()
        return self._conflict_row(row, detail=True) if row else None

    def has_open_conflict(self, document_id: int) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM conflicts WHERE document_id=? AND status='open' LIMIT 1",
                (document_id,),
            ).fetchone()
        return row is not None

    @staticmethod
    def _conflict_row(row: sqlite3.Row, *, detail: bool) -> dict[str, Any]:
        item = dict(row)
        item["metrics"] = json.loads(item.pop("metrics_json"))
        if detail:
            item["diff"] = json.loads(item.pop("diff_json"))
        return item

    def claim_conflict_resolution(self, conflict_id: str, resolution: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE conflicts SET resolution=? WHERE id=? AND status='open' AND resolution IS NULL",
                (f"pending:{resolution}", conflict_id),
            )
        return bool(cursor.rowcount)

    def release_conflict_resolution(self, conflict_id: str, resolution: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE conflicts SET resolution=NULL WHERE id=? AND status='open' AND resolution=?",
                (conflict_id, f"pending:{resolution}"),
            )

    def resolve_conflict(self, conflict_id: str, resolution: str) -> bool:
        now = time.time()
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE conflicts SET status='resolved',resolution=?,resolved_at=?,"
                "existing_text='',generated_text='' "
                "WHERE id=? AND status='open' AND resolution=?",
                (resolution, now, conflict_id, f"pending:{resolution}"),
            )
        return bool(cursor.rowcount)

    def add_decision(
        self,
        *,
        job_id: str,
        document_id: int,
        document_title: str,
        proposal: dict[str, Any],
        applied: dict[str, Any],
        rationale: dict[str, Any],
        before: dict[str, Any],
        status: str,
    ) -> str:
        decision_id = str(uuid.uuid4())
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO decisions(id,job_id,document_id,document_title,proposal_json,applied_json,"
                "rationale_json,before_json,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    decision_id,
                    job_id,
                    document_id,
                    document_title,
                    _json(proposal),
                    _json(applied),
                    _json(rationale),
                    _json(before),
                    status,
                    time.time(),
                ),
            )
        return decision_id

    def list_decisions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id,job_id,document_id,document_title,applied_json,status,created_at "
                "FROM decisions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._decision_summary_row(row) for row in rows]

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM decisions WHERE id=?", (decision_id,)
            ).fetchone()
        return self._decision_row(row) if row else None

    @staticmethod
    def _decision_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for name in ("proposal", "applied", "rationale", "before"):
            item[name] = json.loads(item.pop(f"{name}_json"))
        return item

    @staticmethod
    def _decision_summary_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        applied = json.loads(item.pop("applied_json"))
        item["applied"] = {
            "patch_fields": sorted((applied.get("patch") or {}).keys()),
            "created_count": len(applied.get("created") or []),
            "reused_count": len(applied.get("reused") or []),
            "removed_count": len(applied.get("removed") or []),
        }
        return item

    def set_document_state(
        self,
        document_id: int,
        paperless_modified: str | None,
        source_hash: str | None,
        job_id: str,
        status: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO document_state(document_id,paperless_modified,source_hash,last_job_id,last_status,processed_at) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(document_id) DO UPDATE SET paperless_modified=excluded.paperless_modified,"
                "source_hash=excluded.source_hash,last_job_id=excluded.last_job_id,last_status=excluded.last_status,"
                "processed_at=excluded.processed_at",
                (document_id, paperless_modified, source_hash, job_id, status, time.time()),
            )

    def document_states(self, document_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
        ids = list(document_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM document_state WHERE document_id IN ({placeholders})", ids
            ).fetchall()
        return {row["document_id"]: dict(row) for row in rows}

    def update_document_state_status(self, document_id: int, status: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE document_state SET last_status=?,processed_at=? WHERE document_id=?",
                (status, time.time(), document_id),
            )

    def dashboard_counts(self) -> dict[str, int]:
        with self.connect() as connection:
            jobs = {
                row["status"]: row["count"]
                for row in connection.execute(
                    "SELECT status,COUNT(*) AS count FROM jobs GROUP BY status"
                ).fetchall()
            }
            open_conflicts = connection.execute(
                "SELECT COUNT(*) FROM conflicts WHERE status='open'"
            ).fetchone()[0]
            non_conflict_review = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='needs_review' AND phase != 'ocr_conflict'"
            ).fetchone()[0]
            decisions_today = connection.execute(
                "SELECT COUNT(*) FROM decisions WHERE created_at >= ?", (time.time() - 86_400,)
            ).fetchone()[0]
        return {
            "active": sum(jobs.get(status, 0) for status in ACTIVE_STATUSES),
            "queued": jobs.get("queued", 0) + jobs.get("retry_wait", 0),
            "completed": jobs.get("completed", 0),
            "failed": jobs.get("failed", 0),
            # OCR conflicts have their own count and must not appear twice in
            # the dashboard's intervention total.
            "needs_review": non_conflict_review,
            "open_conflicts": open_conflicts,
            "decisions_today": decisions_today,
        }
