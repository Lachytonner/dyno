"""Search strategy for finding the fastest llama.cpp config."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from collections.abc import Callable

from .bench import (
    extract_model_metadata,
    find_bench_binary,
    run_bench,
)
from .types import BenchParams, TrialResult, TuneResult, score_trial

console = Console()


@dataclass
class TuneConfig:
    mode: str = "quick"  # "quick" or "thorough"
    max_trials: int = 25
    pp: int = 512
    tg: int = 128
    timeout_per_trial: int = 300
    convergence_threshold: float = 0.02  # 2% variance = converged (adaptive budget)

    @classmethod
    def quick(cls) -> TuneConfig:
        return cls(mode="quick", max_trials=10, timeout_per_trial=180)

    @classmethod
    def thorough(cls) -> TuneConfig:
        return cls(mode="thorough", max_trials=25, timeout_per_trial=300)


def _score_trial(t: TrialResult, pp_weight: float = 0.3, tg_weight: float = 0.7) -> float:
    """Score a trial via the canonical scorer in types.score_trial."""
    return score_trial(t, pp_weight, tg_weight)


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
    table.add_column("±", justify="right")
    table.add_column("score", justify="right")
    table.add_column("status", justify="center")

    for i, t in enumerate(trials, 1):
        p = t.params
        score = _score_trial(t)
        if t.oom:
            status = "[red]OOM[/]"
        elif t.error:
            status = "[red]ERR[/]"
        else:
            status = "[green]OK[/]"

        # Calculate stability indicator: larger of pp_stddev/tg_stddev
        if t.pp_stddev is not None and t.tg_stddev is not None:
            stability = max(t.pp_stddev, t.tg_stddev)
        elif t.tg_stddev is not None:
            stability = t.tg_stddev
        elif t.pp_stddev is not None:
            stability = t.pp_stddev
        else:
            stability = None
        stability_str = f"±{stability:.1f}" if stability is not None else "-"

        table.add_row(
            str(i),
            str(p.ngl),
            "✓" if p.flash_attn else "✗",
            f"{p.ct_k}/{p.ct_v}",
            str(p.batch_size),
            str(p.threads) if p.threads > 0 else "auto",
            f"{t.pp_tokens_s:.1f}" if t.pp_tokens_s else "-",
            f"{t.tg_tokens_s:.1f}" if t.tg_tokens_s else "-",
            stability_str,
            f"{score:.1f}" if score >= 0 else "-",
            status,
        )
    return table


def _run_trial(
    model_path: str,
    params: BenchParams,
    binary: str | None,
    config: TuneConfig,
    runner: Callable | None = None,
) -> TrialResult:
    """Run a single trial and return result."""
    _run = runner or run_bench
    result = _run(
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
    pp_weight: float = 0.3,
    tg_weight: float = 0.7,
    is_moe_metadata: bool = False,
    ik_flags: dict | None = None,
    runner: Callable | None = None,
    sweep_fa: bool = True,
    sweep_kv: bool = True,
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

    fa_values = [True]
    if sweep_fa and config.mode == "thorough":
        fa_values = [True, False]
    kv_quants = ["f16"]
    if sweep_kv and config.mode == "thorough":
        kv_quants = ["f16", "q8_0", "q4_0"]

    # Try high-impact combos first
    param_combos = list(itertools.product(ngl_values, fa_values, kv_quants))

    # Cap combos
    if config.mode == "quick":
        param_combos = param_combos[:5]

    # Track best per-ngl for pruning
    ngl_best_scores: dict[int, float] = {}
    previous_ngl: int | None = None

    # Iterate grouped by ngl value for pruning
    for ngl_val in ngl_values:
        if len(trials) >= config.max_trials:
            break

        # Get combos for this ngl value
        combos_for_ngl = [(n, f, k) for n, f, k in param_combos if n == ngl_val]
        best_score_for_ngl = -1.0

        for _, fa_val, kv_q in combos_for_ngl:
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
            if is_ik and is_moe_metadata:
                if not ik_flags or ik_flags.get("fmoe", True):
                    params.fmoe = True
                if not ik_flags or ik_flags.get("rtr", True):
                    params.rtr = True
                if not ik_flags or ik_flags.get("amb", True):
                    params.amb = True

            # Skip if trial with identical params already exists
            if any(t.params == params for t in trials):
                continue

            result = _run_trial(model_path, params, binary, config, runner=runner)
            trials.append(result)

            console.clear()
            console.print(build_progress_table(trials))

            score = _score_trial(result, pp_weight, tg_weight)
            if score > best_score_for_ngl:
                best_score_for_ngl = score
            if score > best_score:
                best_score = score
                best_params = params.clone()

        ngl_best_scores[ngl_val] = best_score_for_ngl

        # Pruning: if current ngl's best score is significantly worse than previous,
        # or significantly better (suggesting we've passed the sweet spot), skip remaining
        if previous_ngl is not None and best_score_for_ngl > 0 and ngl_best_scores.get(previous_ngl, 0) > 0:
            ratio = best_score_for_ngl / ngl_best_scores[previous_ngl]
            if ratio < 0.8:
                console.print(f"[dim]Pruning: ngl={ngl_val} score {ratio:.0%} of ngl={previous_ngl}, skipping remaining ngl values[/]")
                break
            if ratio > 1.2:
                console.print(f"[dim]Pruning: ngl={ngl_val} score {ratio:.0%} of ngl={previous_ngl}, skipping lower ngl values (sweet spot passed)[/]")
                break

        previous_ngl = ngl_val

    return best_params


def _hill_climb(
    model_path: str,
    base_params: BenchParams,
    binary: str | None,
    config: TuneConfig,
    trials: list[TrialResult],
    is_ik: bool,
    pp_weight: float = 0.3,
    tg_weight: float = 0.7,
    is_moe_metadata: bool = False,
    ik_flags: dict | None = None,
    runner: Callable | None = None,
) -> BenchParams:
    """Phase 2: Hill-climb on batch size and threads.

    Starting from the best params found in phase 1, try nearby values
    of batch size and thread count.
    """
    if base_params is None:
        base_params = BenchParams(pp=config.pp, tg=config.tg)

    best_params = base_params.clone()
    best_score = _score_trial(trials[-1], pp_weight, tg_weight) if trials else -1.0

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
    improved = True
    stale_trials = 0
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

            result = _run_trial(model_path, test_params, binary, config, runner=runner)
            trials.append(result)

            console.clear()
            console.print(build_progress_table(trials))

            score = _score_trial(result, pp_weight, tg_weight)

            # Plateau: if score hasn't improved by >1% over best, count stale
            if score <= best_score * 1.01:
                stale_trials += 1
            else:
                stale_trials = 0

            if stale_trials >= 3:
                console.print(f"[dim]Batch plateau: {stale_trials} trials without >1% improvement, stopping batch search[/]")
                break

            if score > best_score:
                best_score = score
                best_params.batch_size = bs
                best_params.ubatch_size = test_params.ubatch_size
                improved = True

    # Hill climb on threads
    improved = True
    stale_trials = 0
    auto_score: float | None = None
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

            result = _run_trial(model_path, test_params, binary, config, runner=runner)
            trials.append(result)

            console.clear()
            console.print(build_progress_table(trials))

            score = _score_trial(result, pp_weight, tg_weight)

            # Track auto-score for thread pruning
            if tc == 0:
                auto_score = score

            # Pruning: if manual thread count gives worse result than auto, skip remaining
            if tc != 0 and auto_score is not None and score < auto_score:
                console.print(f"[dim]Thread pruning: tc={tc} ({score:.1f}) < auto ({auto_score:.1f}), skipping remaining thread counts[/]")
                break

            # Plateau: if score hasn't improved by >1% over best, count stale
            if score <= best_score * 1.01:
                stale_trials += 1
            else:
                stale_trials = 0

            if stale_trials >= 3:
                console.print(f"[dim]Thread plateau: {stale_trials} trials without >1% improvement, stopping thread search[/]")
                break

            if score > best_score:
                best_score = score
                best_params.threads = tc
                improved = True

    # If ik_llama.cpp on an MoE model, try fmoe/rtr/amb toggles
    if is_ik and is_moe_metadata:
        if config.mode == "thorough":
            ik_toggle_combos = [(True, True, True), (True, True, False),
                                (True, False, True), (False, False, False)]
        else:
            ik_toggle_combos = [(True, True, True), (False, False, False)]
        for fmoe_val, rtr_val, amb_val in ik_toggle_combos:
            if len(trials) >= config.max_trials:
                break
            test_params = best_params.clone()
            test_params.fmoe = fmoe_val
            test_params.rtr = rtr_val
            test_params.amb = amb_val

            # Skip if trial with identical params already exists
            if any(t.params == test_params for t in trials):
                continue

            result = _run_trial(model_path, test_params, binary, config, runner=runner)
            trials.append(result)

            console.clear()
            console.print(build_progress_table(trials))

            score = _score_trial(result, pp_weight, tg_weight)
            if score > best_score:
                best_score = score
                best_params.fmoe = fmoe_val
                best_params.rtr = rtr_val
                best_params.amb = amb_val

    return best_params


def _detect_vram_mib() -> int:
    """Detect GPU memory in MiB (NVIDIA VRAM, then Apple unified memory). 0 if unknown.

    A single monkeypatchable seam for the tuner's memory heuristic.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        vram = pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024 * 1024)
        pynvml.nvmlShutdown()
        return int(vram)
    except Exception:
        pass
    import subprocess as sp
    try:
        out = sp.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.stdout.strip():
            return int(out.stdout.strip())
    except Exception:
        pass
    from .detect import _apple_silicon_gpu
    apple = _apple_silicon_gpu()
    return apple[1] if apple else 0


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
    pp_weight: float = 0.3,
    tg_weight: float = 0.7,
    is_ik: bool = False,
    ik_flags: dict | None = None,
    runner: Callable | None = None,
    sweep_fa: bool = True,
    sweep_kv: bool = True,
) -> TuneResult:
    """Run the full tuning pipeline on a model.

    Args:
        model_path: Path to .gguf file.
        mode: "quick" (~10 trials) or "thorough" (~25 trials).
        trials: Optional list to collect results (for testing).
        pp_weight: Weight for prompt-processing throughput (default 0.3).
        tg_weight: Weight for text-generation throughput (default 0.7).

    Returns:
        TuneResult with winning params and full trial list.
    """
    config = TuneConfig.quick() if mode == "quick" else TuneConfig.thorough()
    if trials is None:
        trials = []

    binary = find_bench_binary() if runner is None else None
    if binary is None and runner is None:
        console.print("[red]ERROR: llama-bench not found in PATH.[/]")
        console.print("Install llama.cpp: [bold]brew install llama.cpp[/]")
        console.print("Or build from source: [bold]https://github.com/ggml-org/llama.cpp[/]")
        return TuneResult(
            winning_params=BenchParams(),
            trials=trials,
        )

    # Determine backend
    is_ik = is_ik or (binary is not None and "ik" in binary.lower())

    # Extract model metadata (skip for custom runners like Ollama)
    is_moe_metadata = False
    if runner is None and binary is not None:
        metadata = extract_model_metadata(model_path)
        if metadata.get("build_commit"):
            console.print(f"  Build commit: {metadata['build_commit']}")
        if metadata.get("model_n_params"):
            console.print(f"  Model params: {metadata['model_n_params']}")
        if metadata.get("model_type"):
            console.print(f"  Model type: {metadata['model_type']}")
        if metadata.get("model_size"):
            size_gib = metadata['model_size'] / (1024 ** 3)
            console.print(f"  Model size (llama-bench): {size_gib:.2f} GiB")
        console.print()

        is_moe_metadata = metadata.get("is_moe", False)
        if is_moe_metadata:
            console.print("  Model type: MoE (enabling MoE-specific tuning)")
        else:
            console.print("  Model type: Dense (skipping MoE-specific tuning)")
        console.print()

    vram_total = _detect_vram_mib()
    model_size_mib = _estimate_model_size(model_path) if runner is None else 0

    console.print(f"[bold]Dyno Tuning[/] - mode: {mode}")
    console.print(f"  Model: {model_path}")
    if runner is None:
        console.print(f"  Model size: {model_size_mib} MiB")
    console.print(f"  VRAM: {vram_total} MiB")
    console.print(f"  Backend: {'ik_llama.cpp' if is_ik else 'llama.cpp' if runner is None else 'ollama'}")
    if is_ik and ik_flags:
        parts = []
        if ik_flags.get("fmoe"):
            parts.append("fmoe=✓")
        if ik_flags.get("rtr"):
            parts.append("rtr=✓")
        if ik_flags.get("amb"):
            parts.append("amb=✓")
        if parts:
            console.print(f"  ik_llama.cpp extras: {' '.join(parts)}")
    console.print()

    # Phase 1: Coarse sweep
    console.print("[bold]Phase 1:[/] Coarse sweep (ngl, flash attention, KV cache)...")
    best_params = _coarse_sweep_ngl(
        model_path, vram_total, model_size_mib, binary, config, trials, is_ik,
        pp_weight=pp_weight, tg_weight=tg_weight,
        is_moe_metadata=is_moe_metadata, ik_flags=ik_flags,
        runner=runner, sweep_fa=sweep_fa, sweep_kv=sweep_kv,
    )

    if best_params is None:
        # Everything OOM'd; try minimal config
        console.print("[yellow]All trials OOM'd. Trying minimal config...[/]")
        best_params = BenchParams(ngl=0, flash_attn=False, pp=config.pp, tg=config.tg)
        result = _run_trial(model_path, best_params, binary, config, runner=runner)
        trials.append(result)
        score = _score_trial(result, pp_weight, tg_weight)
        if score < 0:
            return TuneResult(winning_params=best_params, trials=trials)

    # Adaptive budget: check for early convergence after Phase 1
    valid_scores = [
        _score_trial(t, pp_weight, tg_weight)
        for t in trials if not t.oom and not t.error
    ]
    valid_scores.sort(reverse=True)
    top3 = valid_scores[:3]
    if len(top3) >= 2:
        max_s = max(top3)
        min_s = min(top3)
        if max_s > 0 and (max_s - min_s) / max_s <= config.convergence_threshold:
            console.print("[yellow]Config converged early — scores within threshold, skipping hill climb[/]")
            # Skip Phase 2 entirely, go to final scoring
        else:
            # Phase 2: Hill climb
            console.print("[bold]Phase 2:[/] Hill climb (batch size, threads)...")
            best_params = _hill_climb(
                model_path, best_params, binary, config, trials, is_ik,
                pp_weight=pp_weight, tg_weight=tg_weight,
                is_moe_metadata=is_moe_metadata, ik_flags=ik_flags,
                runner=runner,
            )
    else:
        # Too few valid trials, still do hill climb
        console.print("[bold]Phase 2:[/] Hill climb (batch size, threads)...")
        best_params = _hill_climb(
            model_path, best_params, binary, config, trials, is_ik,
            pp_weight=pp_weight, tg_weight=tg_weight,
            runner=runner,
        )

    # Find best trial
    best_score = -1.0
    for t in trials:
        s = _score_trial(t, pp_weight, tg_weight)
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
