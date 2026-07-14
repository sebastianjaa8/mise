"""A durable, SQLite-backed job queue for catalog/index refresh work.

Rebuilding item embeddings + the FAISS index takes real time and shouldn't
block a request thread — a real system would use SQS/Kafka/Redis Streams for
this; SQLite gets the same core property (durable, survives a process
restart, safe under polling) without pulling in a broker dependency for a
single-node demo. Swapping the backend later means replacing this file, not
the callers.
"""
import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Job:
    id: int
    job_type: str
    payload: dict
    status: str
    created_at: float


class JobQueue:
    def __init__(self, db_path="artifacts/jobs.db"):
        self.db_path = db_path
        # FastAPI runs sync endpoint functions in a threadpool, so this
        # connection can be touched from a different thread than the one
        # that created it — check_same_thread=False + SQLite's own internal
        # locking is enough for job-queue-sized traffic (not a hot path).
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL,
                claimed_at REAL,
                completed_at REAL,
                result TEXT,
                error TEXT
            )
        """)
        self._conn.commit()

    def enqueue(self, job_type: str, payload: Optional[dict] = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO jobs (job_type, payload, status, created_at) VALUES (?, ?, 'pending', ?)",
            (job_type, json.dumps(payload or {}), time.time()),
        )
        self._conn.commit()
        return cur.lastrowid

    def claim_next(self) -> Optional[Job]:
        """Claim the oldest pending job. Not concurrency-safe across
        multiple worker processes (would need `UPDATE ... RETURNING` with
        SERIALIZABLE isolation or a broker's native visibility-timeout) —
        fine for a single local worker, flagged as the upgrade path."""
        row = self._conn.execute(
            "SELECT id, job_type, payload, status, created_at FROM jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        job_id = row[0]
        self._conn.execute("UPDATE jobs SET status = 'running', claimed_at = ? WHERE id = ?", (time.time(), job_id))
        self._conn.commit()
        return Job(id=job_id, job_type=row[1], payload=json.loads(row[2]), status="running", created_at=row[4])

    def mark_done(self, job_id: int, result: Optional[dict] = None):
        self._conn.execute(
            "UPDATE jobs SET status = 'done', completed_at = ?, result = ? WHERE id = ?",
            (time.time(), json.dumps(result or {}), job_id),
        )
        self._conn.commit()

    def mark_failed(self, job_id: int, error: str):
        self._conn.execute(
            "UPDATE jobs SET status = 'failed', completed_at = ?, error = ? WHERE id = ?",
            (time.time(), error, job_id),
        )
        self._conn.commit()

    def get(self, job_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT id, job_type, status, created_at, completed_at, result, error FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(zip(["id", "job_type", "status", "created_at", "completed_at", "result", "error"], row))
