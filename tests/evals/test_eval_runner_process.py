"""Subprocess lifecycle and terminal-status tests for the evaluation runner."""

from __future__ import annotations

import io

from remediation_engine.evals import runner


class FakeProcess:
    def __init__(self, output: str, exit_code: int | None = None) -> None:
        self.stdout = io.StringIO(output)
        self.returncode = exit_code
        self.terminated = False
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = -15 if self.terminated else 0
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def test_runner_reports_typed_nonzero_exit(monkeypatch) -> None:
    process = FakeProcess("one line\n", exit_code=3)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    outcomes: list[runner.EvalProcessOutcome] = []
    stream = runner.run_eval_subprocess("report", on_complete=outcomes.append)

    output = list(stream)

    assert any("exit code 3" in line for line in output)
    assert outcomes == [runner.EvalProcessOutcome(exit_code=3)]
    assert outcomes[0].succeeded is False
    assert process.stdout.closed


def test_offline_runner_overrides_ambient_live_mode(monkeypatch) -> None:
    process = FakeProcess("one line\n", exit_code=0)
    captured_command: list[str] = []
    captured_environment: dict[str, str] = {}

    def popen(*args, **kwargs):
        captured_command.extend(args[0])
        captured_environment.update(kwargs["env"])
        return process

    monkeypatch.setenv("EVAL_RUN_LIVE", "true")
    monkeypatch.setattr(runner.subprocess, "Popen", popen)

    list(runner.run_eval_subprocess("report", is_live=False))

    assert captured_environment["EVAL_RUN_LIVE"] == "false"
    assert captured_environment["EVAL_JUDGE_MODEL"] == "gpt-6-luna"
    assert captured_environment["EVAL_JUDGE_REASONING_EFFORT"] == "max"
    assert "--run-eval-live" not in captured_command


def test_runner_exposes_curated_end_to_end_suite(monkeypatch) -> None:
    process = FakeProcess("done\n", exit_code=0)
    captured_command: list[str] = []

    def popen(*args, **kwargs):
        captured_command.extend(args[0])
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    list(runner.run_eval_subprocess("end_to_end", is_live=False))

    assert captured_command[-1] == "tests/evals/test_end_to_end_eval.py"
    assert "--run-eval-live" not in captured_command


def test_live_runner_can_score_an_explicit_end_to_end_trajectory(monkeypatch) -> None:
    process = FakeProcess("done\n", exit_code=0)
    captured_command: list[str] = []

    def popen(*args, **kwargs):
        captured_command.extend(args[0])
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    list(
        runner.run_eval_subprocess(
            "end_to_end",
            is_live=True,
            extra_args=["--e2e-trajectory", "/tmp/complete-run.md"],
        )
    )

    assert "--run-eval-live" in captured_command
    assert "--e2e-trajectory" in captured_command
    assert captured_command[-1] == "tests/evals/test_end_to_end_eval.py"


def test_closing_runner_stream_terminates_process_and_closes_pipe(monkeypatch) -> None:
    process = FakeProcess("first line\n")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    stream = runner.run_eval_subprocess("report")

    next(stream)  # launch notice
    assert next(stream) == "first line\n"
    stream.close()

    assert process.terminated
    assert process.stdout.closed


def test_closing_runner_kills_process_when_termination_times_out(monkeypatch) -> None:
    class StuckProcess(FakeProcess):
        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None:
                raise runner.subprocess.TimeoutExpired("pytest", timeout)
            return super().wait(timeout)

    process = StuckProcess("first line\n")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    stream = runner.run_eval_subprocess("report")
    next(stream)
    next(stream)

    stream.close()

    assert process.terminated
    assert process.killed
    assert process.stdout.closed
