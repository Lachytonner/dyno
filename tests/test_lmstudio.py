"""Offline tests for the LM Studio runner.

Monkeypatches urllib.request.urlopen to avoid real HTTP calls.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from io import BytesIO

import pytest

from conftest import MockBenchRunner

import llama_dyno.lmstudio as lm_mod
import llama_dyno.tune as tune
from llama_dyno.lmstudio import (
    list_lmstudio_models,
    lmstudio_available,
    lmstudio_runner,
    run_lmstudio_bench,
)
from llama_dyno.tune import run_tune
from llama_dyno.types import BenchParams, TrialResult


class FakeResponse(BytesIO):
    def __init__(self, data: bytes, status: int = 200):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _make_sse_chunks(deltas: list[str], usage: dict | None = None) -> bytes:
    lines = []
    if deltas:
        lines.append(
            'data: {"choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}'
        )
        for d in deltas:
            lines.append(
                f'data: {{"choices":[{{"delta":{{"content":{json.dumps(d)}}},"index":0,"finish_reason":null}}]}}'
            )
    finish_data = {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}
    if usage:
        finish_data["usage"] = usage
    lines.append(f"data: {json.dumps(finish_data)}")
    lines.append("data: [DONE]")
    return "\n\n".join(lines).encode() + b"\n\n"


def _make_urlopen(
    models_body: dict | None = None,
    sse_deltas: list[str] | None = None,
    sse_usage: dict | None = None,
):
    models_bytes = json.dumps(models_body if models_body is not None else {"data": []}).encode()
    sse_deltas = sse_deltas if sse_deltas is not None else ["hello"]
    sse_bytes = _make_sse_chunks(deltas=sse_deltas, usage=sse_usage)

    def fake_urlopen(req: urllib.request.Request, timeout: int = 5) -> FakeResponse:
        path = req.get_full_url()
        if "/chat/completions" in path:
            return FakeResponse(sse_bytes)
        if "/models" in path:
            return FakeResponse(models_bytes)
        return FakeResponse(b"{}")

    return fake_urlopen


def _patch_urlopen(
    monkeypatch,
    models_body: dict | None = None,
    sse_deltas: list[str] | None = None,
    sse_usage: dict | None = None,
):
    fake = _make_urlopen(models_body, sse_deltas, sse_usage)
    monkeypatch.setattr(urllib.request, "urlopen", fake)


def _patch_monotonic(monkeypatch, ttft: float = 0.1, gen_window: float = 0.5):
    """Control time.monotonic so we get deterministic tok/s.

    Provides time values for t0 (0.0), first_token_time (ttft), and each
    subsequent last_token_time call (ttft + gen_window + tiny increment).
    """
    call_count: list[int] = [0]

    def fake_monotonic():
        i = call_count[0]
        call_count[0] += 1
        if i == 0:
            return 0.0
        if i == 1:
            return ttft
        return ttft + gen_window + (i - 2) * 0.001

    monkeypatch.setattr(time, "monotonic", fake_monotonic)


def test_lmstudio_available_success(monkeypatch):
    _patch_urlopen(monkeypatch, models_body={"data": [{"id": "llama-3.2-3b"}]})
    assert lmstudio_available("http://localhost:1234/v1") is True


def test_lmstudio_available_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    assert lmstudio_available("http://localhost:1234/v1") is False


def test_list_lmstudio_models(monkeypatch):
    _patch_urlopen(monkeypatch, models_body={
        "data": [{"id": "llama-3.2-3b"}, {"id": "mistral-7b"}]
    })
    models = list_lmstudio_models("http://localhost:1234/v1")
    assert models == ["llama-3.2-3b", "mistral-7b"]


def test_run_lmstudio_bench_parses_tok_per_sec(monkeypatch):
    _patch_urlopen(monkeypatch, sse_deltas=["hello", " world", "!"], sse_usage={
        "prompt_tokens": 512, "completion_tokens": 3
    })
    _patch_monotonic(monkeypatch, ttft=0.2, gen_window=0.4)
    pp, tg = run_lmstudio_bench(
        "llama-3.2-3b", prompt_tokens=512, gen_tokens=128,
        base="http://localhost:1234/v1",
    )
    assert pp is not None
    assert tg is not None
    assert pp == pytest.approx(512 / 0.2, rel=0.1)
    assert tg == pytest.approx(3 / 0.4, rel=0.1)


def test_run_lmstudio_bench_usage_completion_tokens(monkeypatch):
    _patch_urlopen(monkeypatch, sse_deltas=["hello"], sse_usage={
        "prompt_tokens": 512, "completion_tokens": 128
    })
    _patch_monotonic(monkeypatch, ttft=0.1, gen_window=0.5)
    pp, tg = run_lmstudio_bench(
        "llama-3.2-3b", base="http://localhost:1234/v1",
    )
    assert tg == pytest.approx(128 / 0.5, rel=0.1)


def test_run_lmstudio_bench_no_tokens_returns_none(monkeypatch):
    _patch_urlopen(monkeypatch, sse_deltas=[], sse_usage={
        "prompt_tokens": 512, "completion_tokens": 0
    })
    pp, tg = run_lmstudio_bench(
        "llama-3.2-3b", base="http://localhost:1234/v1",
    )
    assert pp is None
    assert tg is None


def test_run_lmstudio_bench_http_error_returns_none(monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.URLError("timeout")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    pp, tg = run_lmstudio_bench(
        "llama-3.2-3b", base="http://localhost:1234/v1",
    )
    assert pp is None
    assert tg is None


def test_lmstudio_runner_returns_trial_result(monkeypatch):
    _patch_urlopen(monkeypatch, sse_deltas=["hello", " world"], sse_usage={
        "prompt_tokens": 512, "completion_tokens": 2
    })
    _patch_monotonic(monkeypatch, ttft=0.1, gen_window=0.3)
    params = BenchParams(pp=512, tg=128)
    result = lmstudio_runner("llama-3.2-3b", params=params)
    assert isinstance(result, TrialResult)
    assert result.pp_tokens_s is not None
    assert result.tg_tokens_s is not None


def test_lmstudio_runner_oom_when_both_none(monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    result = lmstudio_runner("llama-3.2-3b", params=BenchParams())
    assert result.oom is True
    assert result.error is not None


def test_run_tune_with_lmstudio_runner(monkeypatch):
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

    assert len(result.trials) > 0
    assert result.winning_params is not None
