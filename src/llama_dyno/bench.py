"""Run llama-bench and parse results."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

from .detect import _find_binary
from .types import BenchParams, TrialResult


def find_bench_binary() -> str | None:
    """Find llama-bench binary, trying both llama.cpp and ik_llama.cpp names."""
    for name in ["ik_llama-bench", "llama-bench"]:
        found = _find_binary(name)
        if found:
            return found
    return None


def find_server_binary() -> str | None:
    for name in ["ik_llama-server", "llama-server"]:
        found = _find_binary(name)
        if found:
            return found
    return None


def validate_model(model_path: str) -> bool:
    """Check that a model file exists and is a .gguf file."""
    if not os.path.isfile(model_path):
        return False
    if not model_path.lower().endswith(".gguf"):
        return False
    return True


def validate_model_loads(model_path: str, binary: str | None = None) -> bool:
    """Quick check that the model can be loaded by llama-bench."""
    if binary is None:
        binary = find_bench_binary()
    if binary is None:
        return False

    try:
        result = subprocess.run(
            [binary, "-m", model_path, "-p", "1", "-n", "1", "-ngl", "1",
             "--output-json"],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _parse_bench_output(text: str) -> list[dict]:
    """Parse llama-bench JSON output.

    llama-bench with --output-json produces an array of result objects.
    Also handle the older CSV output as fallback.
    """
    # Try JSON first
    text = text.strip()
    if text.startswith("[") or text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            return [data]
        except json.JSONDecodeError:
            pass

    # Try CSV fallback
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("|") or line.startswith("+") or \
           line.startswith("model") or line.startswith("| model"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 8:
            result = {}
            try:
                headers = ["model", "size", "params", "backend", "n_gpu_layers",
                           "main_gpu", "test", "pp", "tg"]
                for i, h in enumerate(headers):
                    if i < len(parts):
                        result[h] = parts[i]
                results.append(result)
            except (ValueError, IndexError):
                continue

    # Try parsing table format (pipe-delimited)
    if not results:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("|"):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 8 and not any(
                    p.startswith("model") or p == "---" for p in parts
                ):
                    try:
                        results.append({
                            "model": parts[0],
                            "size": parts[1],
                            "params": parts[2],
                            "backend": parts[3],
                            "n_gpu_layers": parts[4],
                            "test": parts[5],
                            "pp": parts[6],
                            "tg": parts[7],
                        })
                    except IndexError:
                        continue

    return results


def _extract_from_results(data: list[dict]) -> tuple[float | None, float | None]:
    """Extract pp and tg tokens/sec from parsed bench results.

    llama-bench JSON outputs one entry per test type:
    - Entry with n_prompt>0, n_gen=0 → prompt processing (avg_ts = tok/s)
    - Entry with n_prompt=0, n_gen>0 → text generation (avg_ts = tok/s)
    """
    pp = None
    tg = None

    for entry in data:
        n_prompt = entry.get("n_prompt", 0)
        n_gen = entry.get("n_gen", 0)
        avg_ts = entry.get("avg_ts")

        if avg_ts is not None:
            if n_prompt > 0 and n_gen == 0:
                pp = avg_ts
            elif n_gen > 0 and n_prompt == 0:
                tg = avg_ts

    return pp, tg


def _extract_timing_from_table(text: str) -> tuple[float | None, float | None]:
    """Fallback: extract tokens/s from the table output by parsing the pp/tg columns."""
    pp = None
    tg = None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 8:
            test_col = parts[5]
            pp_col = parts[6]
            tg_col = parts[7]
            if "pp" in test_col or test_col in ("512", "512/128"):
                try:
                    m = re.search(r"([\d.]+)\s*token/s", pp_col)
                    if m:
                        pp = float(m.group(1))
                except (ValueError, IndexError):
                    pass
            if "tg" in test_col or test_col in ("128", "512/128"):
                try:
                    m = re.search(r"([\d.]+)\s*token/s", tg_col)
                    if m:
                        tg = float(m.group(1))
                except (ValueError, IndexError):
                    pass
    return pp, tg


def run_bench(
    model_path: str,
    params: BenchParams | None = None,
    binary: str | None = None,
    timeout: int = 300,
) -> TrialResult:
    """Run a single llama-bench trial with the given parameters.

    Returns a TrialResult with parsed metrics or OOM/error status.
    """
    if params is None:
        params = BenchParams()

    if binary is None:
        binary = find_bench_binary()
    if binary is None:
        return TrialResult(
            params=params,
            oom=False,
            error=(
                "llama-bench not found in PATH.\n"
                "Install llama.cpp: brew install llama.cpp\n"
                "Or build from source: https://github.com/ggml-org/llama.cpp"
            ),
        )

    cmd = [binary, "-m", model_path, "-o", "json"] + params.to_flag_list()

    try:
        start = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.monotonic() - start

    except FileNotFoundError:
        return TrialResult(
            params=params,
            oom=False,
            error=f"Binary not found: {binary}",
        )
    except subprocess.TimeoutExpired:
        return TrialResult(
            params=params,
            oom=True,
            error="Timed out (likely OOM or hang)",
        )
    except OSError as e:
        return TrialResult(
            params=params,
            oom=False,
            error=str(e),
        )

    output = result.stdout

    # Check for OOM conditions also in stderr
    oom_keywords = [
        "CUDA error", "out of memory", "OOM", "cudaMalloc failed",
        "failed to allocate", "Not enough memory", "signal 9",
        "Killed", "terminate called after throwing",
    ]
    is_oom = result.returncode != 0 and any(
        kw.lower() in (result.stdout + result.stderr).lower() for kw in oom_keywords
    )

    if result.returncode != 0 and not is_oom:
        stderr_lines = result.stderr.splitlines()
        prefixed = "\n".join(f"  > {line}" for line in stderr_lines)
        return TrialResult(
            params=params,
            oom=False,
            error=f"llama-bench failed (exit {result.returncode})\n{prefixed}",
        )

    if is_oom:
        return TrialResult(
            params=params,
            oom=True,
            error=result.stderr[:300],
        )

    # Parse output
    output = result.stdout
    data = _parse_bench_output(output)
    pp_ts, tg_ts = _extract_from_results(data)

    # Fallback: try extracting from raw output
    if pp_ts is None or tg_ts is None:
        pp_ts, tg_ts = _extract_timing_from_table(output)

    # Also try scanning for "token/s" anywhere in output
    if pp_ts is None or tg_ts is None:
        for line in output.splitlines():
            m = re.search(r"pp\s*=\s*([\d.]+)\s*token/s", line)
            if m:
                pp_ts = float(m.group(1))
            m = re.search(r"tg\s*=\s*([\d.]+)\s*token/s", line)
            if m:
                tg_ts = float(m.group(1))
            if pp_ts is not None and tg_ts is not None:
                break

    return TrialResult(
        params=params,
        pp_tokens_s=pp_ts,
        tg_tokens_s=tg_ts,
        oom=False,
    )


def get_reproducible_command(model_path: str, params: BenchParams) -> str:
    """Generate the exact command used for a benchmark, for reproducibility."""
    binary = find_bench_binary() or "llama-bench"
    flags = params.to_flag_list()
    cmd = [binary, "-m", model_path] + flags
    return " ".join(cmd)
