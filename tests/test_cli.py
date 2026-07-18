"""CLI integration tests for llama-dyno."""

from __future__ import annotations

from typer.testing import CliRunner

from llama_dyno.cli import app

runner = CliRunner()


def test_cli_detect_help():
    result = runner.invoke(app, ["detect", "--help"])
    assert result.exit_code == 0
    assert "Fingerprint" in result.stdout


def test_cli_tune_help():
    result = runner.invoke(app, ["tune", "--help"])
    assert result.exit_code == 0
    assert "GPU" in result.stdout


def test_cli_bench_help():
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0
    assert "median" in result.stdout


def test_cli_report_help():
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0
    assert "shareable report" in result.stdout


def test_cli_submit_help():
    result = runner.invoke(app, ["submit", "--help"])
    assert result.exit_code == 0
    assert "Submit" in result.stdout


def test_cli_search_help():
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    assert "search" in result.stdout


def test_cli_compare_help():
    result = runner.invoke(app, ["compare", "--help"])
    assert result.exit_code == 0
    assert "compare" in result.stdout


def test_cli_doctor_help():
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout


def test_cli_no_args():
    """Running dyno with no args should show help."""
    result = runner.invoke(app, [])
    assert "Usage" in result.stdout or "Commands" in result.stdout


def test_cli_invalid_model():
    """Tune with non-existent file should error gracefully."""
    result = runner.invoke(app, ["tune", "/nonexistent/model.gguf"])
    assert result.exit_code != 0
