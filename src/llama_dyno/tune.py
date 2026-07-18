"""Search strategy for finding the fastest llama.cpp config."""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, field, asdict
from typing import Callable

from rich.console import Console
from rich.live import Live
from rich.table import Table

from .bench import find_bench_binary, run_bench
from .types import BenchParams, TrialResult, TuneResult

console = Console()


@dataclass
class TuneConfig:
    mode: str = "quick"  # "quick" or "thorough"
    max_trials: int = 25
    pp: int = 512
    tg: int = 128
    timeout_per_trial: int = 300

    @classmethod
    def quick(cls) -> TuneConfig:
        return cls(mode="quick", max_trials=10, timeout_per_trial=180)

    @classmethod
    def thorough(cls) -> TuneConfig:
        return cls(mode="thorough", max_trials=25, timeout_per_trial=300)


def _score_trial(t: TrialResult) -> float:
    """Combined score: 30% pp throughput, 70% tg throughput."""
    if t.oom or t.error:
        return -1.0
    pp = t.pp_tokens_s or 0
    tg = t.tg_tokens_s or 0
    return pp * 0.3 + tg * 0.7


def build_progress_table(trials: list[TrialResult]) -> Table:
    """Build a Rich table showing trial progress."""
    table = Table(title="Dyno - Tuning Progress", box=None)
    table.add_column("#", style="dim")
    table.add_column("ngl", justify="right")
    table.add_column("fa", justify="center")
    table.add_column("ctk/ctv", justify="center")
    table.add_column("batch", justify="right")
    table.add_column("threads", justify="right")
    table.add_column("pp t/s", justify="right")
    table.add_column("tg t/s", justify="right")
    table.add_column("score", justify="right")
    table.add_column("status", justify="center")

    for i, t in enumerate(trials, 1):
        p = t.params
        score = _score_trial(t)
        if t.oom:
            status = "[red]OOM[/]"
        elif t.error:
            status = f"[red]ERR[/]"
        else:
            status = "[green]OK[/]"

        table.add_row(
            str(i),
            str(p.ngl),
            "✓" if p.flash_attn else "✗",
            f"{p.ct_k}/{p.ct_v}",
            str(p.batch_size),
            str(p.threads) if p.threads > 0 else "auto",
            f"{t.pp_tokens_s:.1f}" if t.pp_tokens_s else "-",
            f"{t.tg_tokens_s:.1f}" if t.tg_tokens_s else "-",
            f"{score:.1f}" if score >= 0 else "-",
            status,
        )
    return table


def _run_trial(
    model_path: str,
    params: BenchParams,
    binary: str | None,
    config: TuneConfig,
) -> TrialResult:
    """Run a single trial and return result."""
    result = run_bench(
        model_path=model_path,
        params=params,
        binary=binary,
        timeout=config.timeout_per_trial,
    )
    return result


def _coarse_sweep_ngl(
    model_path: str,
    vram_total_mib: int,
    model_size_mib: int,
    binary: str | None,
    config: TuneConfig,
    trials: list[TrialResult],
    is_ik: bool,
) -> BenchParams | None:
    """Phase 1: Find max working ngl level and initial good params.

    Strategy: start at max offload (99), back off if OOM. Then try flash attention
    and KV cache quant variations.
    """
    best_params = None
    best_score = -1.0

    # Determine initial ngl based on VRAM vs model size heuristic
    # Each layer is roughly model_params_B / n_layers * 2 bytes in fp16
    # For a typical model, each offloaded layer uses ~ model_size / total_layers
    # Simple heuristic: if model fits in VRAM, try max offload
    if model_size_mib > 0 and vram_total_mib > 0:
        # Rough: model needs ~1.2x file size in VRAM with full offload
        vram_ratio = vram_total_mib / model_size_mib if model_size_mib > 0 else 0
    else:
        vram_ratio = 10  # Assume plenty of VRAM

    ngl_values: list[int]
    if config.mode == "quick":
        if vram_ratio > 1.5:
            ngl_values = [99]  # Try full GPU offload
        elif vram_ratio > 0.8:
            ngl_values = [99, 50, 25]  # Progressive backoff
        else:
            ngl_values = [50, 25, 99]  # Start conservative
    else:
        if vram_ratio > 1.5:
            ngl_values = [99, 50]
        elif vram_ratio > 0.8:
            ngl_values = [99, 50, 25]
        else:
            ngl_values = [50, 25, 99, 10]

    fa_values = [True, False] if config.mode == "thorough" else [True]
    kv_quants = ["f16"]
    if config.mode == "thorough":
        kv_quants = ["f16", "q8_0", "q4_0"]

    # Try high-impact combos first
    param_combos = list(itertools.product(ngl_values, fa_values, kv_quants))

    # Cap combos
    if config.mode == "quick":
        param_combos = param_combos[:5]

    for ngl_val, fa_val, kv_q in param_combos:
        if len(trials) >= config.max_trials:
            break

        params = BenchParams(
            ngl=ngl_val,
            flash_attn=fa_val,
            ct_k=kv_q,
            ct_v=kv_q,
            pp=config.pp,
            tg=config.tg,
        )
        if is_ik:
            params.fmoe = True
            params.rtr = True
            params.amb = True

        # Skip if trial with identical params already exists
        if any(t.params == params for t in trials):
            continue

        result = _run_trial(model_path, params, binary, config)
        trials.append(result)

        console.clear()
        console.print(build_progress_table(trials))

        score = _score_trial(result)
        if score > best_score:
            best_score = score
            best_params = params.clone()

        # If full offload OOM'd, reduce further ngl attempts
        if result.oom and ngl_val == 99 and vram_ratio > 1.5:
            # Try less aggressive next
            pass

    return best_params


def _hill_climb(
    model_path: str,
    base_params: BenchParams,
    binary: str | None,
    config: TuneConfig,
    trials: list[TrialResult],
    is_ik: bool,
) -> BenchParams:
    """Phase 2: Hill-climb on batch size and threads.

    Starting from the best params found in phase 1, try nearby values
    of batch size and thread count.
    """
    if base_params is None:
        base_params = BenchParams(pp=config.pp, tg=config.tg)

    best_params = base_params.clone()
    best_score = _score_trial(trials[-1]) if trials else -1.0

    # Batch size search space
    if config.mode == "quick":
        batch_sizes = [256, 512, 1024, 2048]
    else:
        batch_sizes = [128, 256, 512, 1024, 2048, 4096]

    # Thread search space
    import os
    total_cores = os.cpu_count() or 8
    if config.mode == "quick":
        thread_counts = [0, total_cores // 2, total_cores]
    else:
        thread_counts = [0, total_cores // 4, total_cores // 2, total_cores]

    # Hill climb on batch first
    best_batch = best_params.batch_size
    improved = True
    while improved and len(trials) < config.max_trials:
        improved = False
        for bs in batch_sizes:
            if len(trials) >= config.max_trials:
                break
            test_params = best_params.clone()
            test_params.batch_size = bs
            test_params.ubatch_size = min(bs, 512)

            # Skip if trial with identical params already exists
            if any(t.params == test_params for t in trials):
                continue

            result = _run_trial(model_path, test_params, binary, config)
            trials.append(result)

            console.clear()
            console.print(build_progress_table(trials))

            score = _score_trial(result)
            if score > best_score:
                best_score = score
                best_params.batch_size = bs
                best_params.ubatch_size = test_params.ubatch_size
                best_batch = bs
                improved = True

    # Hill climb on threads
    improved = True
    while improved and len(trials) < config.max_trials:
        improved = False
        for tc in thread_counts:
            if len(trials) >= config.max_trials:
                break
            test_params = best_params.clone()
            test_params.threads = tc

            # Skip if trial with identical params already exists
            if any(t.params == test_params for t in trials):
                continue

            result = _run_trial(model_path, test_params, binary, config)
            trials.append(result)

            console.clear()
            console.print(build_progress_table(trials))

            score = _score_trial(result)
            if score > best_score:
                best_score = score
                best_params.threads = tc
                improved = True

    # If ik_llama.cpp, try moe/rtr/amb toggles
    if is_ik and config.mode == "thorough":
        for fmoe_val, rtr_val, amb_val in [(True, True, True), (True, True, False),
                                            (True, False, True), (False, False, False)]:
            if len(trials) >= config.max_trials:
                break
            test_params = best_params.clone()
            test_params.fmoe = fmoe_val
            test_params.rtr = rtr_val
            test_params.amb = amb_val

            # Skip if trial with identical params already exists
            if any(t.params == test_params for t in trials):
                continue

            result = _run_trial(model_path, test_params, binary, config)
            trials.append(result)

            console.clear()
            console.print(build_progress_table(trials))

            score = _score_trial(result)
            if score > best_score:
                best_score = score
                best_params.fmoe = fmoe_val
                best_params.rtr = rtr_val
                best_params.amb = amb_val

    return best_params


def _estimate_model_size(model_path: str) -> int:
    """Estimate model file size in MiB."""
    import os
    try:
        return os.path.getsize(model_path) // (1024 * 1024)
    except OSError:
        return 0


def run_tune(
    model_path: str,
    mode: str = "quick",
    trials: list[TrialResult] | None = None,
) -> TuneResult:
    """Run the full tuning pipeline on a model.

    Args:
        model_path: Path to .gguf file.
        mode: "quick" (~10 trials) or "thorough" (~25 trials).
        trials: Optional list to collect results (for testing).

    Returns:
        TuneResult with winning params and full trial list.
    """
    config = TuneConfig.quick() if mode == "quick" else TuneConfig.thorough()
    if trials is None:
        trials = []

    binary = find_bench_binary()
    if binary is None:
        console.print("[red]ERROR: llama-bench not found in PATH.[/]")
        console.print("Install llama.cpp: [bold]brew install llama.cpp[/]")
        console.print("Or build from source: [bold]https://github.com/ggml-org/llama.cpp[/]")
        return TuneResult(
            winning_params=BenchParams(),
            trials=trials,
        )

    # Determine backend
    is_ik = "ik" in binary.lower()

    # Detect GPU VRAM
    vram_total = 0
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        vram_total = pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024 * 1024)
        pynvml.nvmlShutdown()
    except Exception:
        pass
    # Fallback nvidia-smi
    if vram_total == 0:
        import subprocess as sp
        try:
            out = sp.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            vram_total = int(out.stdout.strip()) if out.stdout.strip() else 0
        except Exception:
            pass

    model_size_mib = _estimate_model_size(model_path)

    console.print(f"[bold]Dyno Tuning[/] - mode: {mode}")
    console.print(f"  Model: {model_path}")
    console.print(f"  Model size: {model_size_mib} MiB")
    console.print(f"  VRAM: {vram_total} MiB")
    console.print(f"  Backend: {'ik_llama.cpp' if is_ik else 'llama.cpp'}")
    console.print()

    # Phase 1: Coarse sweep
    console.print("[bold]Phase 1:[/] Coarse sweep (ngl, flash attention, KV cache)...")
    best_params = _coarse_sweep_ngl(
        model_path, vram_total, model_size_mib, binary, config, trials, is_ik
    )

    if best_params is None:
        # Everything OOM'd; try minimal config
        console.print("[yellow]All trials OOM'd. Trying minimal config...[/]")
        best_params = BenchParams(ngl=0, flash_attn=False, pp=config.pp, tg=config.tg)
        result = _run_trial(model_path, best_params, binary, config)
        trials.append(result)
        score = _score_trial(result)
        if score < 0:
            return TuneResult(winning_params=best_params, trials=trials)

    # Phase 2: Hill climb
    console.print("[bold]Phase 2:[/] Hill climb (batch size, threads)...")
    best_params = _hill_climb(
        model_path, best_params, binary, config, trials, is_ik
    )

    # Find best trial
    best_score = -1.0
    for t in trials:
        s = _score_trial(t)
        if s > best_score:
            best_score = s
            best_params = t.params.clone()

    return TuneResult(
        winning_params=best_params,
        trials=trials,
    )


def run_bench_final(
    model_path: str,
    params: BenchParams,
    n_runs: int = 3,
) -> tuple[float, float, float, float]:
    """Run the winning config N times and return median + variance."""
    pp_results = []
    tg_results = []

    binary = find_bench_binary()
    if binary is None:
        return 0.0, 0.0, 0.0, 0.0

    for i in range(n_runs):
        result = run_bench(model_path, params, binary)
        if result.pp_tokens_s is not None:
            pp_results.append(result.pp_tokens_s)
        if result.tg_tokens_s is not None:
            tg_results.append(result.tg_tokens_s)

    def median(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2.0

    def variance(vals: list[float], med: float) -> float:
        if len(vals) < 2:
            return 0.0
        return sum((v - med) ** 2 for v in vals) / len(vals)

    med_pp = median(pp_results)
    med_tg = median(tg_results)
    var_pp = variance(pp_results, med_pp) if len(pp_results) > 1 else 0.0
    var_tg = variance(tg_results, med_tg) if len(tg_results) > 1 else 0.0

    return med_pp, med_tg, var_pp, var_tg
