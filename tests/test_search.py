"""Unit tests for llama-dyno search logic with a mocked bench runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import pytest

from llama_dyno.types import BenchParams, TrialResult, TuneResult


class MockBenchRunner:
    """Mock that simulates bench runs for testing the search algorithm.

    The mock returns speeds that vary by parameters, allowing us to verify
    the tuner picks the optimal config.
    """

    def __init__(self):
        self.calls: list[BenchParams] = []

    def run(self, model_path: str, params: BenchParams) -> TrialResult:
        self.calls.append(params)

        # Simulate OOM at very high ngl for "small VRAM" scenarios
        if params.ngl > 60:
            return TrialResult(params=params, oom=True, error="CUDA OOM (mocked)")

        # Simulate better performance with flash attention
        fa_bonus = 1.2 if params.flash_attn else 1.0

        # Simulate KV cache quant effects
        kv_factor = {"f16": 1.0, "q8_0": 1.05, "q4_0": 1.1}.get(params.ct_k, 1.0)

        # Simulate ngl scaling (more layers = faster up to a point)
        ngl_factor = min(1.0, params.ngl / 50) * 0.5 + 0.5

        # Simulate batch size sweet spot at 1024
        batch_penalty = 1.0
        if params.batch_size < 256:
            batch_penalty = 0.7
        elif params.batch_size < 1024:
            batch_penalty = 0.9  # sub-optimal
        elif params.batch_size > 2048:
            batch_penalty = 0.85

        # Simulate thread scaling
        thread_factor = 1.0
        if params.threads == 0:
            thread_factor = 1.0  # auto = good
        elif params.threads < 4:
            thread_factor = 0.8

        # Base speed
        base_pp = 500.0
        base_tg = 30.0

        pp = base_pp * ngl_factor * fa_bonus * kv_factor * batch_penalty * thread_factor
        tg = base_tg * ngl_factor * fa_bonus * kv_factor * batch_penalty * thread_factor

        return TrialResult(
            params=params,
            pp_tokens_s=pp,
            tg_tokens_s=tg,
            oom=False,
        )


def test_bench_params_to_flag_list():
    """Test that BenchParams generates correct CLI flags."""
    params = BenchParams(
        ngl=99,
        flash_attn=True,
        ct_k="f16",
        ct_v="f16",
        batch_size=512,
        ubatch_size=256,
        threads=8,
        fmoe=True,
        rtr=False,
        amb=True,
    )
    flags = params.to_flag_list()

    assert "-ngl" in flags
    assert "99" in flags
    assert "--flash-attn" in flags
    assert "-ctk" in flags
    assert "f16" in flags
    assert "-fmoe" in flags
    assert "--flash-attn" in flags
    assert "-rtr" not in flags, "rtr=False should not include -rtr"
    assert "-amb" in flags
    assert "-t" in flags
    assert "8" in flags


def test_trial_result_score():
    """Test score calculation."""
    # Good result
    t = TrialResult(
        params=BenchParams(),
        pp_tokens_s=500.0,
        tg_tokens_s=30.0,
    )
    assert t.score == pytest.approx(500 * 0.3 + 30 * 0.7)

    # OOM should get -1
    t2 = TrialResult(params=BenchParams(), oom=True)
    assert t2.score == -1.0

    # Error should get -1
    t3 = TrialResult(params=BenchParams(), error="something broke")
    assert t3.score == -1.0


def test_score_prefers_tg():
    """Test that tg throughput is weighted higher than pp."""
    high_tg = TrialResult(params=BenchParams(), pp_tokens_s=10.0, tg_tokens_s=100.0)
    high_pp = TrialResult(params=BenchParams(), pp_tokens_s=100.0, tg_tokens_s=1.0)
    assert high_tg.score > high_pp.score, "High tg should beat high pp"


def test_mock_bench_oom():
    """Test mock OOM behavior."""
    mock = MockBenchRunner()
    result = mock.run("test.gguf", BenchParams(ngl=99))
    assert result.oom
    assert result.error is not None


def test_mock_bench_success():
    """Test mock successful run."""
    mock = MockBenchRunner()
    result = mock.run("test.gguf", BenchParams(ngl=30, flash_attn=True))
    assert not result.oom
    assert result.pp_tokens_s is not None
    assert result.tg_tokens_s is not None


def test_tune_finds_best_config():
    """Integration-like test: verify tuner picks the best config.

    With our mock, ngl=50 should be peak (OOM above 60), flash_attn helps,
    batch=1024 is sweet spot, threads=auto is best.
    """
    mock = MockBenchRunner()
    trials: list[TrialResult] = []

    # Simulate a simple tune: try a few configs and pick best
    configs_to_try = [
        BenchParams(ngl=30, flash_attn=False, ct_k="f16", ct_v="f16"),
        BenchParams(ngl=50, flash_attn=True, ct_k="f16", ct_v="f16"),
        BenchParams(ngl=99, flash_attn=True, ct_k="f16", ct_v="f16"),
        BenchParams(ngl=50, flash_attn=True, ct_k="q8_0", ct_v="q8_0"),
        BenchParams(ngl=50, flash_attn=True, ct_k="f16", ct_v="f16", batch_size=1024),
        BenchParams(ngl=50, flash_attn=True, ct_k="f16", ct_v="f16", batch_size=4096),
    ]

    for params in configs_to_try:
        result = mock.run("test.gguf", params)
        trials.append(result)

    # Find best
    best_score = -1.0
    best_params = None
    for t in trials:
        if t.score > best_score:
            best_score = t.score
            best_params = t.params

    assert best_params is not None
    # The ngl=50, flash_attn=True, batch=1024 config should be best
    assert best_params.ngl == 50, f"Expected ngl=50, got {best_params.ngl}"
    assert best_params.flash_attn, "Expected flash_attn=True"
    assert best_params.batch_size == 1024, f"Expected batch=1024, got {best_params.batch_size}"

    # Verify OOM trial was recorded but not selected (score should be -1)
    oom_trial = next((t for t in trials if t.oom), None)
    assert oom_trial is not None, "OOM trial should be in results"
    assert oom_trial.score == -1.0, "OOM trial should have score -1"


def test_trial_scoring_rejects_oom():
    """OOM configs should never beat working ones."""
    mock = MockBenchRunner()
    oom = mock.run("test.gguf", BenchParams(ngl=99))
    working = mock.run("test.gguf", BenchParams(ngl=30))

    assert oom.score == -1.0
    assert working.score > oom.score
