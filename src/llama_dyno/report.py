"""Report generation: JSON, markdown, and reproducible command."""

from __future__ import annotations

import json
from typing import Any

from .bench import get_reproducible_command
from .detect import detect_hardware
from .ollama import ollama_reproducible_command
from .types import (
    HardwareFingerprint,
    TuneResult,
    default_report,
    model_info_from_path,
)


def build_report(
    model_path: str,
    tune_result: TuneResult,
    hardware: HardwareFingerprint | None = None,
    dyno_version: str = "0.1.0",
    backend: str | None = None,
) -> dict[str, Any]:
    """Build a complete report dict."""
    if hardware is None:
        hardware = detect_hardware()

    model_info = model_info_from_path(model_path)

    report = default_report(
        model=model_info,
        hardware=hardware,
        tune_result=tune_result,
        dyno_version=dyno_version,
    )

    # Add reproducible command
    if backend == "ollama":
        report["reproducible_command"] = ollama_reproducible_command(
            model_path, tune_result.winning_params
        )
        report["hardware"]["backend"] = "ollama"
    else:
        report["reproducible_command"] = get_reproducible_command(
            model_path, tune_result.winning_params
        )

    # Add all trials
    report["trials"] = []
    for t in tune_result.trials:
        trial_dict = {
            "params": t.params.to_dict(),
            "pp_tokens_per_sec": t.pp_tokens_s,
            "tg_tokens_per_sec": t.tg_tokens_s,
            "oom": t.oom,
            "error": t.error,
        }
        report["trials"].append(trial_dict)

    return report


def format_json(report: dict[str, Any], indent: int = 2) -> str:
    """Format report as indented JSON."""
    return json.dumps(report, indent=indent, default=str)


def format_markdown(report: dict[str, Any]) -> str:
    """Format report as a shareable markdown snippet."""
    hw = report["hardware"]
    model = report["model"]
    wp = report["winning_params"]
    results = report["results"]

    lines = []
    lines.append("## 🏎️ Dyno Benchmark Report")
    lines.append("")
    lines.append(f"**Model:** {model['name']}")
    lines.append(f"**Quantization:** {model.get('quantization') or 'unknown'}")
    lines.append(f"**Backend:** {hw['backend']} ({hw.get('backend_commit') or 'unknown'})")
    lines.append(f"**GPU:** {hw['gpu_name']} ({hw['vram_total_mib']} MiB)")
    lines.append(f"**Driver:** {hw['driver_version']} | **CUDA:** {hw.get('cuda_version') or 'N/A'}")
    lines.append(f"**CPU:** {hw['cpu_name']} ({hw['cpu_cores']} cores)")
    lines.append(f"**Date:** {report['timestamp']}")
    lines.append(f"**Dyno:** v{report['dyno_version']}")
    lines.append("")

    # Results table
    lines.append("### Results")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| Prompt processing | {results['median_pp_tokens_per_sec']:.1f} tok/s |" if results.get("median_pp_tokens_per_sec") else "| Prompt processing | N/A |")
    lines.append(f"| Text generation | {results['median_tg_tokens_per_sec']:.1f} tok/s |" if results.get("median_tg_tokens_per_sec") else "| Text generation | N/A |")

    if results.get("variance_pp") is not None:
        lines.append(f"| PP variance | {results['variance_pp']:.2f} |")
    if results.get("variance_tg") is not None:
        lines.append(f"| TG variance | {results['variance_tg']:.2f} |")

    lines.append(f"| Trials | {report['trial_count']} |")
    lines.append("")

    # Winning config
    lines.append("### Winning Config")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|------:|")
    lines.append(f"| GPU layers (ngl) | {wp['ngl']} |")
    lines.append(f"| Flash attention | {'✓' if wp['flash_attn'] else '✗'} |")
    lines.append(f"| K cache quant | `{wp['ct_k']}` |")
    lines.append(f"| V cache quant | `{wp['ct_v']}` |")
    lines.append(f"| Batch size | {wp['batch_size']} |")
    lines.append(f"| UB batch size | {wp['ubatch_size']} |")
    lines.append(f"| Threads | {wp['threads'] or 'auto'} |")
    if wp.get("fmoe"):
        lines.append(f"| Fast MoE | {'✓' if wp['fmoe'] else '✗'} |")
    if wp.get("rtr"):
        lines.append(f"| Runtime reorder | {'✓' if wp['rtr'] else '✗'} |")
    if wp.get("amb"):
        lines.append(f"| Attn mem bound | {'✓' if wp['amb'] else '✗'} |")
    lines.append("")

    # Reproducible command
    lines.append("### Reproducible Command")
    lines.append("")
    lines.append("```bash")
    lines.append(report.get("reproducible_command", ""))
    lines.append("```")
    lines.append("")

    # Full fingerprint
    lines.append("### Hardware Fingerprint")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(hw, indent=2, default=str))
    lines.append("```")

    return "\n".join(lines)


def save_report_json(report: dict[str, Any], path: str | None = None) -> str:
    """Save report to a JSON file. Returns path.

    If no path is provided, generates a versioned filename in the current
    directory using the same naming scheme as submit._result_filename():
    GPU_MODEL_quant_hash_vX.json
    """
    import os

    if path is None:
        hw = report.get("hardware", {})
        model = report.get("model", {})
        quant = model.get("quantization", "unknown")
        short_hash = model.get("sha256", "unknown")[:12] if model.get("sha256") else "unknown"
        version = report.get("dyno_version", "0.0.0")

        gpu_name = hw.get("gpu_name", "unknown-gpu")
        safe_gpu = "".join(c if c.isalnum() or c in " -_" else "_" for c in gpu_name)
        safe_gpu = safe_gpu.replace(" ", "_").lower()

        model_name = model.get("name", "unknown-model")
        if model_name.endswith(".gguf"):
            model_name = model_name[:-5]
        safe_model = "".join(c if c.isalnum() or c in " -_" else "_" for c in model_name)
        safe_model = safe_model.replace(" ", "_").lower()

        path = f"{safe_gpu}_{safe_model}_{quant}_{short_hash}_v{version}.json"

    resolved = os.path.abspath(path)
    os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
    with open(resolved, "w") as f:
        f.write(format_json(report))
    return resolved
