"""The MCP job registry: one job per project name, run off the request thread."""

import threading

import pytest

from jobs import Jobs, JobsFull


def inline(work):
    work()


def test_a_job_runs_and_is_done():
    jobs = Jobs(submit=inline)
    job = jobs.start("song", lambda: None)
    assert job.status == "done"
    assert job.error is None
    assert jobs.get("song") is job


def test_status_is_running_while_the_work_runs():
    jobs = Jobs(submit=inline)
    seen = []

    def work():
        job = jobs.get("song")
        assert job is not None
        seen.append(job.status)

    jobs.start("song", work)
    assert seen == ["running"]


def test_a_failing_job_records_the_error():
    def refused():
        raise RuntimeError("400: Unsupported file type")

    job = Jobs(submit=inline).start("song", refused)
    assert job.status == "failed"
    assert job.error == "400: Unsupported file type"


def test_an_exception_without_a_message_is_named():
    def bare():
        raise ValueError

    jobs = Jobs(submit=inline)
    assert jobs.start("song", bare).error == "ValueError"


def test_a_name_already_pending_returns_the_same_job():
    queued = []
    jobs = Jobs(submit=queued.append)
    first = jobs.start("song", lambda: None)
    second = jobs.start("song", lambda: None)
    assert second is first
    assert first.status == "queued"
    assert len(queued) == 1


def test_a_failed_name_can_be_started_again():
    def refused():
        raise RuntimeError("502: Bad Gateway")

    jobs = Jobs(submit=inline)
    jobs.start("song", refused)
    again = jobs.start("song", lambda: None)
    assert again.status == "done"
    assert jobs.get("song") is again


def test_pending_jobs_are_capped():
    jobs = Jobs(submit=lambda work: None, max_pending=2)
    jobs.start("a", lambda: None)
    jobs.start("b", lambda: None)
    with pytest.raises(JobsFull):
        jobs.start("c", lambda: None)


def test_check_room_raises_when_full():
    jobs = Jobs(submit=lambda work: None, max_pending=1)
    jobs.start("a", lambda: None)
    with pytest.raises(JobsFull, match="1 analyses pending"):
        jobs.check_room("b")


def test_check_room_passes_for_a_name_already_pending():
    jobs = Jobs(submit=lambda work: None, max_pending=1)
    jobs.start("a", lambda: None)
    jobs.check_room("a")


def test_check_room_passes_when_there_is_room():
    jobs = Jobs(submit=lambda work: None, max_pending=2)
    jobs.start("a", lambda: None)
    jobs.check_room("b")


def test_finished_jobs_do_not_count_against_the_cap():
    jobs = Jobs(submit=inline, max_pending=1)
    jobs.start("a", lambda: None)
    assert jobs.start("b", lambda: None).status == "done"


def test_the_default_pool_runs_jobs_on_another_thread():
    ran = threading.Event()
    threads = []

    def work():
        threads.append(threading.current_thread())
        ran.set()

    Jobs().start("song", work)
    assert ran.wait(5)
    assert threads[0] is not threading.current_thread()
