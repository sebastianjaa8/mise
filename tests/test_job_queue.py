"""Job queue mechanics: enqueue -> claim -> done/failed, and dispatch."""
from mise import worker
from mise.job_queue import JobQueue


def test_enqueue_claim_done_cycle(tmp_path):
    queue = JobQueue(db_path=str(tmp_path / "jobs.db"))

    job_id = queue.enqueue("rebuild_index", {"reason": "catalog updated"})
    assert queue.get(job_id)["status"] == "pending"

    claimed = queue.claim_next()
    assert claimed.id == job_id
    assert claimed.job_type == "rebuild_index"
    assert queue.get(job_id)["status"] == "running"

    assert queue.claim_next() is None  # nothing else pending

    queue.mark_done(job_id, {"catalog_size": 42})
    row = queue.get(job_id)
    assert row["status"] == "done"
    assert row["completed_at"] is not None


def test_mark_failed(tmp_path):
    queue = JobQueue(db_path=str(tmp_path / "jobs.db"))
    job_id = queue.enqueue("rebuild_index")
    queue.claim_next()
    queue.mark_failed(job_id, "boom")
    assert queue.get(job_id)["status"] == "failed"
    assert queue.get(job_id)["error"] == "boom"


def test_process_one_dispatches_to_handler(tmp_path, monkeypatch):
    queue = JobQueue(db_path=str(tmp_path / "jobs.db"))
    calls = []
    monkeypatch.setitem(worker.HANDLERS, "noop", lambda job: calls.append(job.id) or {"ok": True})

    job_id = queue.enqueue("noop")
    assert worker.process_one(queue) is True
    assert calls == [job_id]
    assert queue.get(job_id)["status"] == "done"


def test_process_one_unknown_job_type_fails_cleanly(tmp_path):
    queue = JobQueue(db_path=str(tmp_path / "jobs.db"))
    job_id = queue.enqueue("does_not_exist")
    assert worker.process_one(queue) is True
    assert queue.get(job_id)["status"] == "failed"


def test_process_one_empty_queue_returns_false(tmp_path):
    queue = JobQueue(db_path=str(tmp_path / "jobs.db"))
    assert worker.process_one(queue) is False
