import time
from pathlib import Path

from paperless_clerk.db import Database


def database(tmp_path: Path) -> Database:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    return db


def test_active_job_enqueue_is_idempotent(tmp_path: Path) -> None:
    db = database(tmp_path)

    first, created_first = db.enqueue_job(101, "full", 3)
    second, created_second = db.enqueue_job(101, "ocr", 3)

    assert created_first
    assert not created_second
    assert second["id"] == first["id"]
    assert db.dashboard_counts()["active"] == 1


def test_retry_schedule_and_attempt_limit_are_persisted(tmp_path: Path) -> None:
    db = database(tmp_path)
    queued, _ = db.enqueue_job(102, "full", 2)

    first = db.claim_job("worker-a", 60)
    assert first and first["attempt"] == 1
    assert db.fail_or_retry(first["id"], "timeout", "model timed out", True) == "retry_wait"

    with db.connect() as connection:
        connection.execute(
            "UPDATE jobs SET next_run_at=? WHERE id=?", (time.time() - 1, queued["id"])
        )
    second = db.claim_job("worker-b", 60)
    assert second and second["attempt"] == 2
    assert db.fail_or_retry(second["id"], "timeout", "model timed out again", True) == "failed"
    assert db.get_job(second["id"])["status"] == "failed"


def test_expired_lease_is_reclaimed(tmp_path: Path) -> None:
    db = database(tmp_path)
    queued, _ = db.enqueue_job(103, "full", 3)
    first = db.claim_job("dead-worker", 60)
    assert first
    with db.connect() as connection:
        connection.execute(
            "UPDATE jobs SET lease_until=? WHERE id=?", (time.time() - 1, queued["id"])
        )

    reclaimed = db.claim_job("new-worker", 60)

    assert reclaimed and reclaimed["id"] == first["id"]
    assert reclaimed["attempt"] == 2
    assert reclaimed["worker_id"] == "new-worker"


def test_completed_pages_survive_retry(tmp_path: Path) -> None:
    db = database(tmp_path)
    job, _ = db.enqueue_job(104, "ocr", 3)
    db.upsert_page(job["id"], 1, "complete", 1, text="page one")
    db.upsert_page(job["id"], 2, "failed_retryable", 1, error="timeout")

    rows = db.page_results(job["id"])
    detail = db.get_job(job["id"], include_events=True)

    assert rows[0]["text"] == "page one"
    assert rows[1]["status"] == "failed_retryable"
    assert detail and detail["page_failures"] == [
        {
            "page_number": 2,
            "status": "failed_retryable",
            "attempts": 1,
            "error": "timeout",
            "duration_ms": None,
        }
    ]


def test_ocr_version_checkpoint_survives_claims_and_can_be_cleared(tmp_path: Path) -> None:
    db = database(tmp_path)
    job, _ = db.enqueue_job(107, "ocr", 3)

    db.set_ocr_version_task(job["id"], "paperless-task")
    db.complete_ocr_version(job["id"], 207)

    checkpoint = db.get_job(job["id"])
    assert checkpoint and checkpoint["ocr_version_task_id"] == "paperless-task"
    assert checkpoint["ocr_version_id"] == 207

    db.clear_ocr_version_checkpoint(job["id"])

    cleared = db.get_job(job["id"])
    assert cleared and cleared["ocr_version_task_id"] is None
    assert cleared["ocr_version_id"] is None


def test_ocr_conflict_is_not_double_counted_as_generic_review(tmp_path: Path) -> None:
    db = database(tmp_path)
    job, _ = db.enqueue_job(105, "full", 3)
    db.mark_needs_review(job["id"], "ocr_conflict", "review")
    db.create_conflict(
        job_id=job["id"],
        document_id=105,
        document_title="Conflicted document",
        existing_text="existing text",
        generated_text="generated text",
        score=0.4,
        metrics={"score": 0.4},
        diff=[],
        tag_id=9,
    )

    counts = db.dashboard_counts()

    assert counts["open_conflicts"] == 1
    assert counts["needs_review"] == 0


def test_decision_lists_expose_counts_not_evidence_details(tmp_path: Path) -> None:
    db = database(tmp_path)
    job, _ = db.enqueue_job(106, "metadata", 3)
    decision_id = db.add_decision(
        job_id=job["id"],
        document_id=106,
        document_title="Private decision",
        proposal={"tags": [{"evidence": "sensitive excerpt"}]},
        applied={
            "patch": {"tags": [1]},
            "reused": [{"name": "Medical", "evidence": "sensitive excerpt"}],
            "created": [],
        },
        rationale={"summary": "private"},
        before={},
        status="applied",
    )

    summary = db.list_decisions()[0]
    detail = db.get_job(job["id"], include_events=True)

    assert summary["applied"] == {
        "patch_fields": ["tags"],
        "created_count": 0,
        "reused_count": 1,
        "removed_count": 0,
    }
    assert "sensitive excerpt" not in str(summary)
    assert detail and detail["decision_id"] == decision_id
