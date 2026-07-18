"""Integration tests for the tuning search, run fully offline.

These exercise run_tune and its phases (coarse sweep, hill climb, convergence,
pruning, MoE toggles) by monkeypatching the module's external calls — the mock
bench runner stands in for real llama-bench subprocesses.
"""

from __future__ import annotations

from conftest import MockBenchRunner

import llama_dyno.tune as tune
from llama_dyno.tune import TuneConfig, run_tune
from llama_dyno.types import TrialResult


def _patch(monkeypatch, runner, *, vram=10_000, size=8_000, metadata=None):
    """Wire run_tune's external calls to offline stand-ins.

    vram/size chosen so vram_ratio lands in the 0.8-1.5 band → the coarse sweep
    tries ngl [99, 50, 25]; the mock OOMs above 60, so ngl=50 is the real peak.
    """
    monkeypatch.setattr(tune, "run_bench", runner)
    monkeypatch.setattr(tune, "find_bench_binary", lambda: "fake-bench")
    monkeypatch.setattr(tune, "extract_model_metadata", lambda _p: metadata or {})
    monkeypatch.setattr(tune, "_detect_vram_mib", lambda: vram)
    monkeypatch.setattr(tune, "_estimate_model_size", lambda _p: size)


def test_run_tune_finds_best_config_and_rejects_oom(monkeypatch):
    mock = MockBenchRunner()
    _patch(monkeypatch, mock.run)

    result = run_tune("dummy.gguf", mode="quick")
    wp = result.winning_params

    # Mock optimum: ngl=50 (99 OOMs, 25 is slower), FA on, batch sweet spot 1024.
    assert wp.ngl == 50
    assert wp.flash_attn is True
    assert wp.batch_size == 1024
    assert wp.threads == 0  # auto beats forced thread counts in the mock

    # An OOM config was tried but never selected as the winner.
    assert any(t.oom for t in result.trials), "ngl=99 should have OOM'd"
    winner_trial = next(t for t in result.trials if t.params == wp)
    assert not winner_trial.oom


def test_run_tune_respects_trial_budget(monkeypatch):
    mock = MockBenchRunner()
    _patch(monkeypatch, mock.run)

    result = run_tune("dummy.gguf", mode="thorough")
    assert len(result.trials) <= TuneConfig.thorough().max_trials


def test_run_tune_does_not_rebench_duplicate_params(monkeypatch):
    mock = MockBenchRunner()
    _patch(monkeypatch, mock.run)

    run_tune("dummy.gguf", mode="thorough")

    seen = [tuple(sorted(p.to_dict().items())) for p in mock.calls]
    assert len(seen) == len(set(seen)), "identical configs were benched twice"


def test_run_tune_early_convergence_skips_hill_climb(monkeypatch):
    # Constant, never-OOM runner → Phase-1 scores are identical → converged.
    def const_runner(model_path, params=None, binary=None, timeout=300, warmup=True, **kw):
        return TrialResult(params=params, pp_tokens_s=100.0, tg_tokens_s=20.0)

    _patch(monkeypatch, const_runner)
    result = run_tune("dummy.gguf", mode="quick")

    # Hill climb is the only phase that varies batch size; if it ran, some trial
    # would have batch != 512. Convergence must have skipped it.
    assert all(t.params.batch_size == 512 for t in result.trials)


def test_run_tune_moe_enables_ik_flags(monkeypatch):
    mock = MockBenchRunner()
    _patch(monkeypatch, mock.run, metadata={"is_moe": True})

    result = run_tune("dummy.gguf", mode="quick", is_ik=True)
    assert any(t.params.fmoe for t in result.trials), "MoE + ik should try -fmoe"


def test_run_tune_dense_skips_ik_flags(monkeypatch):
    mock = MockBenchRunner()
    _patch(monkeypatch, mock.run, metadata={"is_moe": False})

    result = run_tune("dummy.gguf", mode="quick", is_ik=True)
    assert not any(t.params.fmoe for t in result.trials), "dense model should skip -fmoe"
