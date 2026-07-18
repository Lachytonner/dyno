"""Offline tests for the Ollama runner.

Monkeypatches urllib.request.urlopen to avoid real HTTP calls.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from io import BytesIO

import pytest

from conftest import MockBenchRunner

import llama_dyno.ollama as ollama_mod
import llama_dyno.tune as tune
from llama_dyno.ollama import (
    list_ollama_models,
    ollama_available,
    ollama_runner,
    run_ollama_bench,
)
from llama_dyno.tune import run_tune
from llama_dyno.types import BenchParams, TrialResult


class FakeResponse(BytesIO):
    """A urllib.response-like object wrapping bytes."""

    def __init__(self, data: bytes, status: int = 200):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _make_urlopen(generate_body: dict | None = None, tags_body: dict | None = None):
    """Build a urlopen stand-in that returns canned JSON for known paths."""
    generate_bytes = json.dumps(generate_body or {}).encode() if generate_body else b"{}"
    tags_bytes = json.dumps(tags_body or {}).encode() if tags_body else b'{"models": []}'

    def fake_urlopen(req: urllib.request.Request, timeout: int = 5) -> FakeResponse:
        path = req.get_full_url()
        if "/api/generate" in path:
            return FakeResponse(generate_bytes)
        if "/api/tags" in path:
            return FakeResponse(tags_bytes)
        return FakeResponse(b"{}")

    return fake_urlopen


def _patch_urlopen(monkeypatch, generate_body: dict | None = None, tags_body: dict | None = None):
    fake = _make_urlopen(generate_body, tags_body)
    monkeypatch.setattr(urllib.request, "urlopen", fake)


def test_ollama_available_success(monkeypatch):
    _patch_urlopen(monkeypatch, tags_body={"models": [{"name": "llama3:8b"}]})
    assert ollama_available("http://localhost:11434") is True


def test_ollama_available_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    assert ollama_available("http://localhost:11434") is False


def test_list_ollama_models(monkeypatch):
    _patch_urlopen(monkeypatch, tags_body={
        "models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]
    })
    models = list_ollama_models("http://localhost:11434")
    assert models == ["llama3:8b", "mistral:7b"]


def test_run_ollama_bench_parses_tok_per_sec(monkeypatch):
    _patch_urlopen(monkeypatch, generate_body={
        "prompt_eval_count": 100,
        "prompt_eval_duration": 500_000_000,
        "eval_count": 50,
        "eval_duration": 1_000_000_000,
    })
    pp, tg = run_ollama_bench(
        "llama3:8b", options={"num_gpu": 99},
        prompt_tokens=512, gen_tokens=128, host="http://localhost:11434",
    )
    assert pp is not None
    assert tg is not None
    assert pp == pytest.approx(100 / 0.5)
    assert tg == pytest.approx(50 / 1.0)


def test_run_ollama_bench_zero_duration_returns_none(monkeypatch):
    _patch_urlopen(monkeypatch, generate_body={
        "prompt_eval_count": 100,
        "prompt_eval_duration": 0,
        "eval_count": 50,
        "eval_duration": 0,
    })
    pp, tg = run_ollama_bench(
        "llama3:8b", options={},
        host="http://localhost:11434",
    )
    assert pp is None
    assert tg is None


def test_run_ollama_bench_http_error_returns_none(monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.URLError("timeout")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    pp, tg = run_ollama_bench("llama3:8b", options={}, host="http://localhost:11434")
    assert pp is None
    assert tg is None


def test_ollama_runner_maps_benchparams_to_options(monkeypatch):
    _patch_urlopen(monkeypatch, generate_body={
        "prompt_eval_count": 512,
        "prompt_eval_duration": 1_000_000_000,
        "eval_count": 128,
        "eval_duration": 2_000_000_000,
    })
    params = BenchParams(
        ngl=50,
        batch_size=1024,
        threads=8,
        pp=512,
        tg=128,
    )
    result = ollama_runner("llama3:8b", params=params)
    assert isinstance(result, TrialResult)
    assert result.pp_tokens_s is not None
    assert result.tg_tokens_s is not None


def test_ollama_runner_oom_when_both_none(monkeypatch):
    _patch_urlopen(monkeypatch, generate_body={
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": 0,
    })
    result = ollama_runner("llama3:8b", params=BenchParams(ngl=99))
    assert result.oom is True
    assert result.error is not None


def test_ollama_runner_injects_num_ctx(monkeypatch):
    captured = {}

    def capture_urlopen(req, timeout=5):
        body = json.loads(req.data.decode())
        captured["body"] = body
        return FakeResponse(json.dumps({
            "prompt_eval_count": 512,
            "prompt_eval_duration": 1_000_000_000,
            "eval_count": 128,
            "eval_duration": 2_000_000_000,
        }).encode())

    monkeypatch.setattr(urllib.request, "urlopen", capture_urlopen)

    params = BenchParams(pp=1024, tg=256)
    ollama_runner("llama3:8b", params=params)
    assert captured["body"]["options"]["num_ctx"] == max(1024 + 256, 512)
    assert captured["body"]["options"]["num_gpu"] == params.ngl
    assert captured["body"]["options"]["num_batch"] == params.batch_size


def test_run_tune_with_ollama_runner_uses_sweep_flags(monkeypatch):
    def no_oom_runner(model_path, params=None, binary=None, timeout=300, warmup=True, **kw):
        p = params or BenchParams()
        return TrialResult(params=p, pp_tokens_s=100.0, tg_tokens_s=20.0)

    monkeypatch.setattr(tune, "find_bench_binary", lambda: "fake-bench")
    monkeypatch.setattr(tune, "extract_model_metadata", lambda _p: {})
    monkeypatch.setattr(tune, "_detect_vram_mib", lambda: 10_000)
    monkeypatch.setattr(tune, "_estimate_model_size", lambda _p: 8_000)

    result = run_tune(
        "dummy.gguf", mode="quick",
        runner=no_oom_runner,
        sweep_fa=False,
        sweep_kv=False,
    )

    all_same_fa = len(set(t.params.flash_attn for t in result.trials)) == 1
    all_same_kv = len(set(t.params.ct_k for t in result.trials)) == 1
    assert all_same_fa, "sweep_fa=False should fix flash_attn to one value"
    assert all_same_kv, "sweep_kv=False should fix ct_k to one value"
