"""Dyno CLI — auto-tune and benchmark llama.cpp inference on NVIDIA GPUs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__ as DYNOVERSION
from .bench import find_bench_binary, find_server_binary, run_bench, validate_model
from .detect import detect_hardware
from .report import build_report, format_json, format_markdown, save_report_json
from .submit import submit_report
from .tune import TuneConfig, build_progress_table, run_bench_final, run_tune
from .types import BenchParams, TuneResult

log = logging.getLogger("dyno")


def setup_logging(level: str = "WARNING") -> None:
    """Configure dyno logging with the given level."""
    logging.basicConfig(
        format="%(levelname)-8s | %(message)s",
        level=getattr(logging, level.upper(), logging.WARNING),
    )


app = typer.Typer(
    name="dyno",
    help="Auto-tune and benchmark llama.cpp / ik_llama.cpp inference on NVIDIA GPUs.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main_options(
    ctx: typer.Context,
    log_level: str = typer.Option(
        "WARNING", "--log-level", "-l",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
        case_sensitive=False,
    ),
) -> None:
    """Configure global options."""
    setup_logging(log_level)


def _truncate_model_path(path: str, max_len: int = 50) -> str:
    """Truncate a model path for display in suggestion hints.

    If the path is shorter than max_len, return it as-is.
    Otherwise show just the filename. If even the filename is too long,
    truncate the middle of the full path with '...'.
    """
    if len(path) <= max_len:
        return path
    name = Path(path).name
    if len(name) <= max_len:
        return name
    half = (max_len - 3) // 2
    return path[:half] + "..." + path[-half:]


def _check_mark(passed: bool, label: str) -> str:
    """Return a Rich-formatted check mark string for doctor output."""
    if passed:
        return f"[green]✓ PASS[/] {label}"
    return f"[red]✗ FAIL[/] {label}"


@app.command()
def doctor(
    model: str = typer.Argument(None, help="Optional model path to validate loading"),
) -> None:
    """Run comprehensive system diagnostics."""
    import json
    import platform as plat_mod
    import subprocess

    passed = 0
    total = 0

    console.print("[bold]Dyno Doctor — System Diagnostics[/]")
    console.print()

    # 1. Python check
    total += 1
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    plat = plat_mod.platform()
    console.print(_check_mark(True, f"Python {py_ver} on {plat}"))
    passed += 1

    # 2. GPU check
    total += 1
    hw = detect_hardware()
    if hw.gpu_name != "Unknown" and hw.vram_total_mib > 0:
        console.print(_check_mark(
            True,
            f"GPU: {hw.gpu_name} ({hw.vram_total_mib} MiB VRAM, "
            f"driver {hw.driver_version}, CUDA {hw.cuda_version or 'N/A'})",
        ))
        passed += 1
    else:
        console.print(_check_mark(False, "No NVIDIA GPU detected"))

    # 3. Binary checks
    total += 1
    bench_bin = find_bench_binary()
    if bench_bin:
        console.print(_check_mark(True, f"llama-bench: {bench_bin}"))
        passed += 1
    else:
        console.print(_check_mark(False, "llama-bench Not found"))

    total += 1
    server_bin = find_server_binary()
    if server_bin:
        console.print(_check_mark(True, f"llama-server: {server_bin}"))
        passed += 1
    else:
        console.print(_check_mark(False, "llama-server Not found"))

    # 4. llama-bench smoke test
    total += 1
    if bench_bin:
        try:
            result = subprocess.run(
                [bench_bin, "-o", "json", "-p", "1", "-n", "1", "-ngl", "0"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                try:
                    json.loads(result.stdout)
                    console.print(_check_mark(True, "llama-bench responds"))
                    passed += 1
                except json.JSONDecodeError:
                    console.print(_check_mark(
                        False,
                        f"llama-bench output not valid JSON: {result.stdout.strip()[:100]}",
                    ))
            else:
                err = (result.stderr or result.stdout)[:200]
                console.print(_check_mark(False, f"llama-bench failed: {err.strip()}"))
        except subprocess.TimeoutExpired:
            console.print(_check_mark(False, "llama-bench timed out"))
        except Exception as e:
            console.print(_check_mark(False, f"llama-bench error: {e}"))
    else:
        console.print(_check_mark(False, "llama-bench not available (skipped)"))

    # 5. Model check (optional)
    if model:
        total += 1
        if validate_model(model):
            size_mib = Path(model).stat().st_size // (1024 * 1024)
            console.print(_check_mark(True, f"Model: {Path(model).name} ({size_mib} MiB)"))
            passed += 1
        else:
            console.print(_check_mark(False, "Model not found or not a .gguf file"))

        # 6. Model load test
        total += 1
        if bench_bin and validate_model(model):
            try:
                result = subprocess.run(
                    [bench_bin, "-m", model, "-p", "1", "-n", "1", "-ngl", "0", "-o", "json"],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0:
                    console.print(_check_mark(True, "Model loads successfully"))
                    passed += 1
                else:
                    err = (result.stderr or result.stdout)[:200]
                    console.print(_check_mark(False, f"Model load failed: {err.strip()}"))
            except subprocess.TimeoutExpired:
                console.print(_check_mark(False, "Model load timed out"))
            except Exception as e:
                console.print(_check_mark(False, f"Model load error: {e}"))
        else:
            console.print(_check_mark(
                False, "Cannot test model load (bench binary or model unavailable)",
            ))

    # Summary
    console.print()
    if passed == total:
        console.print(f"[green]All {passed}/{total} checks passed![/]")
    else:
        console.print(f"[yellow]{passed}/{total} checks passed[/]")


@app.command()
def detect():
    """Fingerprint hardware and detect llama.cpp backend."""
    log.debug("Detecting hardware...")
    hw = detect_hardware()

    table = Table(title="System Detection", box=None)
    table.add_column("Component", style="bold cyan")
    table.add_column("Value")

    table.add_row("GPU", hw.gpu_name)
    table.add_row("VRAM", f"{hw.vram_total_mib} MiB")
    table.add_row("Driver", hw.driver_version)
    table.add_row("CUDA", hw.cuda_version or "N/A")
    table.add_row("CPU", hw.cpu_name)
    table.add_row("Cores", str(hw.cpu_cores))
    table.add_row("RAM", f"{hw.ram_total_mib} MiB")
    table.add_row("Backend", hw.backend)
    table.add_row("Commit", hw.backend_commit or "unknown")

    console.print(table)

    # Also check binary availability
    bench_bin = find_bench_binary()
    server_bin = find_server_binary()

    status = Table(title="Binary Status", box=None)
    status.add_column("Binary", style="bold cyan")
    status.add_column("Status")

    if bench_bin:
        status.add_row("llama-bench", f"✓ [dim]{bench_bin}[/]")
    else:
        status.add_row("llama-bench", "✗ Not found")

    if server_bin:
        status.add_row("llama-server", f"✓ [dim]{server_bin}[/]")
    else:
        status.add_row("llama-server", "✗ Not found")

    if not bench_bin and not server_bin:
        console.print()
        console.print("[yellow]No llama.cpp binaries found. Install:[/]")
        console.print("  brew install llama.cpp")
        console.print("  Or build from: https://github.com/ggml-org/llama.cpp")

    console.print()
    console.print(status)


@app.command()
def tune(
    model: str = typer.Argument(
        ..., help="Path to a .gguf model file", exists=True, dir_okay=False
    ),
    quick: bool = typer.Option(
        False, "--quick", help="Quick mode (~10 trials, for fast iteration)"
    ),
    thorough: bool = typer.Option(
        False, "--thorough", help="Thorough mode (~25 trials, best results)"
    ),
    json_out: str = typer.Option(
        None, "--json-out",
        help="Save tuning report as JSON (suppresses Rich output during tuning)",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", help="Suppress all console output (use with --json-out)"
    ),
    pp_weight: float = typer.Option(
        0.3, "--pp-weight",
        min=0.0, max=1.0,
        help="Weight for prompt-processing throughput in scoring (0.0-1.0)",
    ),
    tg_weight: float = typer.Option(
        0.7, "--tg-weight",
        min=0.0, max=1.0,
        help="Weight for text-generation throughput in scoring (0.0-1.0)",
    ),
    optimize: str = typer.Option(
        None, "--optimize",
        help="Optimization preset (overrides --pp-weight/--tg-weight): "
             "balanced (0.3/0.7), generation (0.1/0.9), prompt (0.7/0.3), interactive (0.5/0.5)",
    ),
):
    """Find the fastest config for this GPU + model combo."""
    log.debug("Starting tune for %s", model)
    if not validate_model(model):
        console.print(f"[red]ERROR:[/] '{model}' is not a valid .gguf file.")
        raise typer.Exit(1)

    # Check binary
    if find_bench_binary() is None:
        console.print("[red]ERROR: llama-bench not found in PATH.[/]")
        console.print("Install: [bold]brew install llama.cpp[/]")
        console.print("Or build from: [bold]https://github.com/ggml-org/llama.cpp[/]")
        raise typer.Exit(1)

    mode = "thorough" if thorough else "quick"

    # Resolve optimize preset (overrides individual weights)
    if optimize is not None:
        presets = {
            "balanced": (0.3, 0.7),
            "generation": (0.1, 0.9),
            "prompt": (0.7, 0.3),
            "interactive": (0.5, 0.5),
        }
        if optimize not in presets:
            console.print(f"[red]ERROR:[/] Unknown --optimize value '{optimize}'. Choose from: {', '.join(presets)}")
            raise typer.Exit(1)
        pp_weight, tg_weight = presets[optimize]

    # Detect hardware to determine backend and ik flags
    hw = detect_hardware()
    is_ik = hw.backend == "ik_llama.cpp"
    ik_flags = getattr(hw, "ik_features", None)

    if json_out:
        # Silence tuning console output
        import llama_dyno.tune as tune_mod
        orig_console = tune_mod.console
        tune_mod.console = Console(quiet=True)
        try:
            result = run_tune(model, mode=mode, pp_weight=pp_weight, tg_weight=tg_weight,
                              is_ik=is_ik, ik_flags=ik_flags)
        finally:
            tune_mod.console = orig_console

        # Build report (hw already detected above)
        if result.trials:
            med_pp, med_tg, var_pp, var_tg = run_bench_final(
                model, result.winning_params, n_runs=3,
            )
            result.median_pp_tokens_s = med_pp
            result.median_tg_tokens_s = med_tg
            result.variance_pp = var_pp
            result.variance_tg = var_tg

        report_data = build_report(model, result, hardware=hw, dyno_version=DYNOVERSION)
        save_report_json(report_data, json_out)

        if not quiet:
            console.print()
            console.print("[bold green]✓ Tuning complete![/]")
            wp = result.winning_params
            console.print(Panel(
                f"[bold]Winning Config:[/]\n"
                f"  ngl={wp.ngl}, fa={'✓' if wp.flash_attn else '✗'}\n"
                f"  ctk={wp.ct_k}, ctv={wp.ct_v}\n"
                f"  batch={wp.batch_size}, ubatch={wp.ubatch_size}\n"
                f"  threads={wp.threads or 'auto'}"
                + (f"\n  fmoe={'✓' if wp.fmoe else '✗'}, rtr={'✓' if wp.rtr else '✗'}, amb={'✓' if wp.amb else '✗'}"
                   if wp.fmoe or wp.rtr or wp.amb else ""),
                title="🏆 Best Config",
            ))
            fa_flag = "--fa" if wp.flash_attn else "--no-fa"
            console.print(f"\n[yellow]Next:[/] [bold]dyno bench {_truncate_model_path(model)} --ngl {wp.ngl} {fa_flag} --ctk {wp.ct_k} --ctv {wp.ct_v} --batch {wp.batch_size} --ubatch {wp.ubatch_size} --threads {wp.threads}[/]")
        return

    result = run_tune(model, mode=mode, pp_weight=pp_weight, tg_weight=tg_weight,
                      is_ik=is_ik, ik_flags=ik_flags)

    if result.trials:
        console.print()
        console.print(build_progress_table(result.trials))

    console.print()
    console.print("[bold green]✓ Tuning complete![/]")

    wp = result.winning_params
    console.print(Panel(
        f"[bold]Winning Config:[/]\n"
        f"  ngl={wp.ngl}, fa={'✓' if wp.flash_attn else '✗'}\n"
        f"  ctk={wp.ct_k}, ctv={wp.ct_v}\n"
        f"  batch={wp.batch_size}, ubatch={wp.ubatch_size}\n"
        f"  threads={wp.threads or 'auto'}"
        + (f"\n  fmoe={'✓' if wp.fmoe else '✗'}, rtr={'✓' if wp.rtr else '✗'}, amb={'✓' if wp.amb else '✗'}"
           if wp.fmoe or wp.rtr or wp.amb else ""),
        title="🏆 Best Config",
    ))

    # Suggest next step
    fa_flag = "--fa" if wp.flash_attn else "--no-fa"
    console.print(f"\n[yellow]Next:[/] [bold]dyno bench {_truncate_model_path(model)} --ngl {wp.ngl} {fa_flag} --ctk {wp.ct_k} --ctv {wp.ct_v} --batch {wp.batch_size} --ubatch {wp.ubatch_size} --threads {wp.threads}[/]")


@app.command()
def bench(
    model: str = typer.Argument(
        ..., help="Path to a .gguf model file", exists=True, dir_okay=False
    ),
    ngl: int = typer.Option(99, "--ngl", "-ngl", help="GPU layers to offload"),
    flash_attn: bool = typer.Option(True, "--fa/--no-fa", help="Flash attention"),
    ctk: str = typer.Option("f16", "--ctk", help="K cache quant (f16, q8_0, q4_0)"),
    ctv: str = typer.Option("f16", "--ctv", help="V cache quant (f16, q8_0, q4_0)"),
    batch: int = typer.Option(512, "--batch", "-b", help="Batch size"),
    ubatch: int = typer.Option(512, "--ubatch", "-ub", help="Micro batch size"),
    threads: int = typer.Option(0, "--threads", "-t", help="Threads (0 = auto)"),
    runs: int = typer.Option(3, "--runs", "-r", help="Number of benchmark runs"),
    fmoe: bool = typer.Option(False, "--fmoe", help="Fast MoE (ik_llama.cpp)"),
    rtr: bool = typer.Option(False, "--rtr", help="Runtime reorder (ik_llama.cpp)"),
    amb: bool = typer.Option(False, "--amb", help="Attn mem bound (ik_llama.cpp)"),
):
    """Run the winning config 3x, report median with variance."""
    if not validate_model(model):
        console.print(f"[red]ERROR:[/] '{model}' is not a valid .gguf file.")
        raise typer.Exit(1)

    if find_bench_binary() is None:
        console.print("[red]ERROR: llama-bench not found in PATH.[/]")
        console.print("Install: [bold]brew install llama.cpp[/]")
        console.print("Or build from: [bold]https://github.com/ggml-org/llama.cpp[/]")
        raise typer.Exit(1)

    params = BenchParams(
        ngl=ngl,
        flash_attn=flash_attn,
        ct_k=ctk,
        ct_v=ctv,
        batch_size=batch,
        ubatch_size=ubatch,
        threads=threads,
        fmoe=fmoe,
        rtr=rtr,
        amb=amb,
    )
    log.debug("Running benchmark with ngl=%d fa=%s ctk=%s ctv=%s batch=%d ubatch=%d threads=%d",
              ngl, flash_attn, ctk, ctv, batch, ubatch, threads)

    console.print(f"[bold]Benchmarking[/] with {runs} runs...")

    med_pp, med_tg, var_pp, var_tg = run_bench_final(model, params, n_runs=runs)

    table = Table(title="Benchmark Results", box=None)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Median", justify="right")
    table.add_column("Variance", justify="right")

    if med_pp:
        table.add_row("PP (prompt processing)", f"{med_pp:.1f} tok/s", f"{var_pp:.2f}" if var_pp else "-")
    else:
        table.add_row("PP (prompt processing)", "N/A", "-")

    if med_tg:
        table.add_row("TG (text generation)", f"{med_tg:.1f} tok/s", f"{var_tg:.2f}" if var_tg else "-")
    else:
        table.add_row("TG (text generation)", "N/A", "-")

    console.print(table)

    if med_tg:
        console.print()
        console.print(f"[green]✓[/] Generation: {med_tg:.1f} tok/s | Prompt: {med_pp:.1f} tok/s")
        console.print(f"  Config: ngl={ngl}, fa={flash_attn}, ctk={ctk}, ctv={ctv}, b={batch}, ub={ubatch}, t={threads}")
        console.print(f"\n[yellow]Next:[/] [bold]dyno report {_truncate_model_path(model)}[/] to generate a shareable report")


@app.command()
def report(
    model: str = typer.Argument(
        ..., help="Path to a .gguf model file", exists=True, dir_okay=False
    ),
    json_output: str = typer.Option(
        None, "--json", "-j", help="Save JSON report to path"
    ),
    from_json: str = typer.Option(
        None, "--from-json",
        help="Load existing JSON report file instead of running hardware detection and tuning",
    ),
    no_tune: bool = typer.Option(
        False, "--no-tune", help="Skip tuning, just detect + bench with defaults"
    ),
    quick: bool = typer.Option(
        False, "--quick", help="Quick tune mode"
    ),
):
    """Generate a shareable report: JSON + markdown snippet + reproducible command."""
    if from_json:
        import json as json_mod
        with open(from_json) as f:
            report_data = json_mod.load(f)

        console.print("[bold]Dyno Report Generator[/]")
        console.print("  Loading from existing JSON report...")
        console.print()
        console.print("[bold]📋 Shareable Report[/]")
        console.print()
        console.print(format_markdown(report_data))
        console.print()
        console.print("[bold]📄 Full JSON[/]")
        console.print()
        console.print(format_json(report_data))
        console.print()

        # Save fresh copy
        hw = report_data.get("hardware", {})
        gpu_name = hw.get("gpu_name", "Unknown")
        default_path = f"dyno-report-{gpu_name.replace(' ', '_')}-{Path(model).stem}.json"
        save_report_json(report_data, default_path)
        console.print(f"[green]✓[/] Report saved to: {default_path}")

        if json_output:
            path = save_report_json(report_data, json_output)
            console.print(f"[green]✓[/] Report also saved to: {json_output}")

        console.print(f"\n[yellow]Next:[/] [bold]dyno submit {_truncate_model_path(model)}[/] to share your results with the community")
        return

    if not validate_model(model):
        console.print(f"[red]ERROR:[/] '{model}' is not a valid .gguf file.")
        raise typer.Exit(1)

    console.print("[bold]Dyno Report Generator[/]")

    # Detect hardware
    console.print("  Detecting hardware...")
    hw = detect_hardware()

    # Tune or use defaults
    if no_tune:
        tune_result = TuneResult(winning_params=BenchParams(), trials=[])
    else:
        mode = "quick" if quick else "thorough"
        console.print(f"  Tuning ({mode} mode)...")
        tune_result = run_tune(model, mode=mode)

    # Final bench
    if tune_result.trials:
        console.print("  Running final benchmark (3 runs)...")
        med_pp, med_tg, var_pp, var_tg = run_bench_final(
            model, tune_result.winning_params, n_runs=3
        )
        tune_result.median_pp_tokens_s = med_pp
        tune_result.median_tg_tokens_s = med_tg
        tune_result.variance_pp = var_pp
        tune_result.variance_tg = var_tg

    # Build report
    report_data = build_report(model, tune_result, hardware=hw, dyno_version=DYNOVERSION)

    # Print markdown
    console.print()
    console.print("[bold]📋 Shareable Report[/]")
    console.print()
    console.print(format_markdown(report_data))
    console.print()

    # Print JSON
    console.print("[bold]📄 Full JSON[/]")
    console.print()
    console.print(format_json(report_data))
    console.print()

    # Save if requested
    if json_output:
        path = save_report_json(report_data, json_output)
        console.print(f"[green]✓[/] Report saved to: {path}")

    # Save to default location too
    default_path = f"dyno-report-{hw.gpu_name.replace(' ', '_')}-{Path(model).stem}.json"
    save_report_json(report_data, default_path)
    console.print(f"[green]✓[/] Report also saved to: {default_path}")

    console.print(f"\n[yellow]Next:[/] [bold]dyno submit {_truncate_model_path(model)}[/] to share your results with the community")


@app.command()
def submit(
    model: str = typer.Argument(
        ..., help="Path to a .gguf model file", exists=True, dir_okay=False
    ),
    quick: bool = typer.Option(
        False, "--quick", help="Quick tune mode before submission"
    ),
):
    """Submit results — creates a Gist with your benchmark."""
    if not validate_model(model):
        console.print(f"[red]ERROR:[/] '{model}' is not a valid .gguf file.")
        raise typer.Exit(1)

    console.print("[bold]Dyno Submit[/]")

    # Detect hardware
    hw = detect_hardware()

    # Tune
    mode = "quick" if quick else "thorough"
    console.print(f"  Tuning ({mode} mode)...")
    tune_result = run_tune(model, mode=mode)

    # Final bench
    if tune_result.trials:
        console.print("  Running final benchmark (3 runs)...")
        med_pp, med_tg, var_pp, var_tg = run_bench_final(
            model, tune_result.winning_params, n_runs=3
        )
        tune_result.median_pp_tokens_s = med_pp
        tune_result.median_tg_tokens_s = med_tg
        tune_result.variance_pp = var_pp
        tune_result.variance_tg = var_tg

    # Build report
    report_data = build_report(model, tune_result, hardware=hw, dyno_version=DYNOVERSION)

    console.print("  Submitting...")
    try:
        url = submit_report(report_data)
        console.print(f"[green]✓[/] Results submitted!")
        console.print(f"   {url}")
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)


@app.command()
def search(
    gpu: str = typer.Argument(
        "", help="GPU name or keyword to search for (e.g. 'RTX 4070')"
    ),
    model: str = typer.Argument(
        "", help="Model name or keyword to search for (e.g. 'gemma')"
    ),
    local_dir: str = typer.Option(
        "dyno-results", "--dir", "-d",
        help="Directory of local result JSON files to search",
    ),
):
    """Search local result JSON files for matching GPU and/or model keywords."""
    import json as json_mod

    search_dir = Path(local_dir)
    if not search_dir.is_dir():
        console.print(f"[yellow]Warning:[/] Directory '{local_dir}' not found.")
        return

    results: list[dict[str, str]] = []
    for json_path in sorted(search_dir.glob("*.json")):
        try:
            with open(json_path) as f:
                data = json_mod.load(f)
        except (json_mod.JSONDecodeError, OSError) as e:
            console.print(f"[yellow]Warning:[/] Skipping '{json_path.name}': {e}")
            continue

        # Check it looks like a dyno report (has hardware/model keys)
        hw = data.get("hardware", {})
        model_info = data.get("model", {})

        gpu_name = hw.get("gpu_name", "") or ""
        model_name = model_info.get("name", "") or ""

        # Apply filters (case-insensitive substring match)
        if gpu and gpu.lower() not in gpu_name.lower():
            continue
        if model and model.lower() not in model_name.lower():
            continue

        # Extract display fields
        wp = data.get("winning_params", {})
        results_data = data.get("results", {})
        quant = model_info.get("quantization") or "N/A"
        backend = hw.get("backend", "N/A")
        tg = results_data.get("median_tg_tokens_per_sec")
        pp = results_data.get("median_pp_tokens_per_sec")

        tg_str = f"{tg:.1f}" if tg is not None else "N/A"
        pp_str = f"{pp:.1f}" if pp is not None else "N/A"

        results.append({
            "gpu": gpu_name,
            "model": model_name,
            "quant": quant,
            "backend": backend,
            "tg": tg_str,
            "pp": pp_str,
            "file": json_path.name,
        })

    if not results:
        console.print(f"No matching results found in '{local_dir}'.")
        return

    table = Table(title=f"Search Results ({len(results)} found)")
    table.add_column("GPU", style="bold cyan")
    table.add_column("Model")
    table.add_column("Quant")
    table.add_column("Backend")
    table.add_column("TG tok/s", justify="right")
    table.add_column("PP tok/s", justify="right")
    table.add_column("File")

    for r in results:
        table.add_row(
            r["gpu"], r["model"], r["quant"], r["backend"],
            r["tg"], r["pp"], r["file"],
        )

    console.print(table)
    console.print(f"[green]Found {len(results)} matching results.[/]")


@app.command()
def compare(
    result1: str = typer.Argument(
        ..., help="First result JSON file", exists=True
    ),
    result2: str = typer.Argument(
        ..., help="Second result JSON file", exists=True
    ),
):
    """Compare two result JSON files side-by-side."""
    import json as json_mod

    def load_result(path: str) -> dict:
        with open(path) as f:
            return json_mod.load(f)

    try:
        r1 = load_result(result1)
    except (json_mod.JSONDecodeError, OSError) as e:
        console.print(f"[red]Error:[/] Could not load '{result1}': {e}")
        raise typer.Exit(1)

    try:
        r2 = load_result(result2)
    except (json_mod.JSONDecodeError, OSError) as e:
        console.print(f"[red]Error:[/] Could not load '{result2}': {e}")
        raise typer.Exit(1)

    def _g(d: dict, *keys: str, default: str = "N/A"):
        """Safely get nested value from dict."""
        val: object = d  # type: ignore[assignment]
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
                if val is None:
                    return default
            else:
                return default
        return val

    def _fmt(val: object) -> str:
        if val is None or val == "N/A":
            return "N/A"
        if isinstance(val, float):
            return f"{val:.1f}"
        if isinstance(val, bool):
            return "✓" if val else "✗"
        return str(val)

    # Build comparison rows
    rows: list[tuple[str, str, str]] = []

    # GPU
    rows.append((
        "GPU",
        _fmt(_g(r1, "hardware", "gpu_name")),
        _fmt(_g(r2, "hardware", "gpu_name")),
    ))
    # Model
    rows.append((
        "Model",
        _fmt(_g(r1, "model", "name")),
        _fmt(_g(r2, "model", "name")),
    ))
    # Quant
    rows.append((
        "Quant",
        _fmt(_g(r1, "model", "quantization")),
        _fmt(_g(r2, "model", "quantization")),
    ))
    # Backend
    rows.append((
        "Backend",
        _fmt(_g(r1, "hardware", "backend")),
        _fmt(_g(r2, "hardware", "backend")),
    ))
    # ngl
    rows.append((
        "ngl",
        _fmt(_g(r1, "winning_params", "ngl")),
        _fmt(_g(r2, "winning_params", "ngl")),
    ))
    # FA
    rows.append((
        "FA",
        _fmt(_g(r1, "winning_params", "flash_attn")),
        _fmt(_g(r2, "winning_params", "flash_attn")),
    ))
    # Batch
    rows.append((
        "Batch",
        _fmt(_g(r1, "winning_params", "batch_size")),
        _fmt(_g(r2, "winning_params", "batch_size")),
    ))
    # TG tok/s (highlight faster one in green)
    tg1 = _g(r1, "results", "median_tg_tokens_per_sec")
    tg2 = _g(r2, "results", "median_tg_tokens_per_sec")
    tg1_s = _fmt(tg1)
    tg2_s = _fmt(tg2)
    if isinstance(tg1, (int, float)) and isinstance(tg2, (int, float)):
        if tg1 > tg2:
            tg1_s = f"[green]{tg1_s}[/]"
        elif tg2 > tg1:
            tg2_s = f"[green]{tg2_s}[/]"
    rows.append(("TG tok/s", tg1_s, tg2_s))
    # PP tok/s (highlight faster one in green)
    pp1 = _g(r1, "results", "median_pp_tokens_per_sec")
    pp2 = _g(r2, "results", "median_pp_tokens_per_sec")
    pp1_s = _fmt(pp1)
    pp2_s = _fmt(pp2)
    if isinstance(pp1, (int, float)) and isinstance(pp2, (int, float)):
        if pp1 > pp2:
            pp1_s = f"[green]{pp1_s}[/]"
        elif pp2 > pp1:
            pp2_s = f"[green]{pp2_s}[/]"
    rows.append(("PP tok/s", pp1_s, pp2_s))
    # Config
    def _config_summary(wp: object) -> str:
        if not isinstance(wp, dict):
            return "N/A"
        parts = []
        if "ngl" in wp:
            parts.append(f"ngl={wp['ngl']}")
        if "flash_attn" in wp:
            parts.append("FA" if wp["flash_attn"] else "no-FA")
        if "batch_size" in wp:
            parts.append(f"batch={wp['batch_size']}")
        if "ct_k" in wp and "ct_v" in wp:
            parts.append(f"ctk={wp['ct_k']},ctv={wp['ct_v']}")
        return ", ".join(parts) if parts else "N/A"
    rows.append((
        "Config",
        _config_summary(_g(r1, "winning_params")),
        _config_summary(_g(r2, "winning_params")),
    ))
    # Command
    rows.append((
        "Command",
        _fmt(_g(r1, "reproducible_command")),
        _fmt(_g(r2, "reproducible_command")),
    ))

    table = Table(title="Comparison")
    table.add_column("Metric", style="bold cyan")
    table.add_column(f"Result 1 ({Path(result1).name})")
    table.add_column(f"Result 2 ({Path(result2).name})")

    for metric, val1, val2 in rows:
        table.add_row(metric, val1, val2)

    console.print(table)



