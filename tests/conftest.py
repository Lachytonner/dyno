"""Shared test fixtures and mocks for llama-dyno."""

from __future__ import annotations

from llama_dyno.types import BenchParams, TrialResult


class MockBenchRunner:
    """Simulates llama-bench runs so the search algorithm can be tested offline.

    Speeds vary by parameter so the tuner has a deterministic optimum to find:
    OOM above ngl 60, flash attention helps, KV quant helps slightly, ngl scales
    up to 50, batch peaks in 1024-2048, and thread=auto is best.

    The signature mirrors bench.run_bench so it can be monkeypatched in as a
    drop-in replacement (positional or keyword args, plus binary/timeout/warmup).
    """

    def __init__(self):
        self.calls: list[BenchParams] = []

    def run(
        self,
        model_path: str,
        params: BenchParams | None = None,
        binary: str | None = None,
        timeout: int = 300,
        warmup: bool = True,
        **kwargs,
    ) -> TrialResult:
        params = params or BenchParams()
        self.calls.append(params)

        # Simulate OOM at very high ngl
        if params.ngl > 60:
            return TrialResult(params=params, oom=True, error="CUDA OOM (mocked)")

        fa_bonus = 1.2 if params.flash_attn else 1.0
        kv_factor = {"f16": 1.0, "q8_0": 1.05, "q4_0": 1.1}.get(params.ct_k, 1.0)
        ngl_factor = min(1.0, params.ngl / 50) * 0.5 + 0.5

        batch_penalty = 1.0
        if params.batch_size < 256:
            batch_penalty = 0.7
        elif params.batch_size < 1024:
            batch_penalty = 0.9
        elif params.batch_size > 2048:
            batch_penalty = 0.85

        thread_factor = 1.0
        if 0 < params.threads < 4:
            thread_factor = 0.8

        base_pp = 500.0
        base_tg = 30.0
        mult = ngl_factor * fa_bonus * kv_factor * batch_penalty * thread_factor
        return TrialResult(
            params=params,
            pp_tokens_s=base_pp * mult,
            tg_tokens_s=base_tg * mult,
            oom=False,
        )
