"""Tests for the submit fallback chain (PR → Gist → local), with gh/git mocked."""

from __future__ import annotations

import os
import subprocess

import pytest

from llama_dyno import submit


@pytest.fixture
def report():
    return {
        "hardware": {"gpu_name": "NVIDIA GeForce RTX 4070", "backend": "llama.cpp"},
        "model": {"name": "test-model.gguf", "sha256": "abc123def4567890", "quantization": "Q4_K_M"},
        "dyno_version": "1.0.2",
    }


def _cp(args, rc=0, out="", err=""):
    return subprocess.CompletedProcess(args, rc, stdout=out, stderr=err)


def test_submit_report_prefers_pr(report, monkeypatch):
    monkeypatch.setattr(submit, "submit_via_pr", lambda r: "https://github.com/x/pull/1")
    monkeypatch.setattr(submit, "submit_via_gist", lambda r: "https://gist.github.com/x")
    assert submit.submit_report(report) == "https://github.com/x/pull/1"


def test_submit_report_falls_back_to_gist(report, monkeypatch):
    monkeypatch.setattr(submit, "submit_via_pr", lambda r: None)
    monkeypatch.setattr(submit, "submit_via_gist", lambda r: "https://gist.github.com/x")
    assert submit.submit_report(report) == "https://gist.github.com/x"


def test_submit_report_falls_back_to_local(report, monkeypatch, tmp_path):
    monkeypatch.setattr(submit, "submit_via_pr", lambda r: None)
    monkeypatch.setattr(submit, "submit_via_gist", lambda r: None)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="saved locally"):
        submit.submit_report(report)

    saved = list((tmp_path / "dyno-results").glob("*.json"))
    assert len(saved) == 1


def test_submit_via_pr_returns_none_without_gh(report, monkeypatch):
    monkeypatch.setattr(submit, "_gh_installed", lambda: False)
    assert submit.submit_via_pr(report) is None


def test_submit_via_pr_success(report, monkeypatch):
    monkeypatch.setattr(submit, "_gh_installed", lambda: True)
    monkeypatch.setattr(submit, "_gh_auth_status", lambda: True)

    def fake_run(cmd, cwd=None):
        if cmd[:3] == ["gh", "repo", "fork"]:
            os.makedirs(os.path.join(cwd, "llama-dyno-results"), exist_ok=True)
            return _cp(cmd, 0)
        if cmd[:3] == ["gh", "api", "user"]:
            return _cp(cmd, 0, out="octocat")
        if cmd[:3] == ["gh", "pr", "create"]:
            return _cp(cmd, 0, out="https://github.com/Lachytonner/llama-dyno-results/pull/7")
        return _cp(cmd, 0)  # git checkout/add/commit/push

    monkeypatch.setattr(submit, "_run", fake_run)
    url = submit.submit_via_pr(report)
    assert url == "https://github.com/Lachytonner/llama-dyno-results/pull/7"


def test_submit_via_pr_returns_none_when_push_fails(report, monkeypatch):
    monkeypatch.setattr(submit, "_gh_installed", lambda: True)
    monkeypatch.setattr(submit, "_gh_auth_status", lambda: True)

    def fake_run(cmd, cwd=None):
        if cmd[:3] == ["gh", "repo", "fork"]:
            os.makedirs(os.path.join(cwd, "llama-dyno-results"), exist_ok=True)
            return _cp(cmd, 0)
        if cmd[:2] == ["git", "push"]:
            return _cp(cmd, 1, err="push rejected")
        return _cp(cmd, 0)

    monkeypatch.setattr(submit, "_run", fake_run)
    assert submit.submit_via_pr(report) is None
