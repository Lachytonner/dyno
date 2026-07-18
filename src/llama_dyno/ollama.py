"""Ollama runner — benchmark Ollama models through its REST API.

Reuses the Dyno search/scoring/report machinery by satisfying the
run_bench signature so it slots into the existing injection seam.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .types import BenchParams, TrialResult

OLLAMA_DEFAULT_HOST = "http://localhost:11434"


def ollama_host(default: str = OLLAMA_DEFAULT_HOST) -> str:
    return os.environ.get("OLLAMA_HOST", default)


def _api_url(host: str, path: str) -> str:
    base = host.rstrip("/")
    return f"{base}{path}"


def ollama_available(host: str | None = None) -> bool:
    host = host or ollama_host()
    try:
        req = urllib.request.Request(_api_url(host, "/api/tags"), method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def list_ollama_models(host: str | None = None) -> list[str]:
    host = host or ollama_host()
    req = urllib.request.Request(_api_url(host, "/api/tags"), method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode())
    return [m["name"] for m in data.get("models", [])]


def _make_prompt(num_tokens: int) -> str:
    words = ["hello", "world", "test", "prompt", "dyno", "benchmark"]
    repeat = max(1, num_tokens // len(words))
    return " ".join(words * repeat)


def run_ollama_bench(
    model: str,
    options: dict[str, Any],
    prompt_tokens: int = 512,
    gen_tokens: int = 128,
    host: str | None = None,
    timeout: int = 300,
) -> tuple[float | None, float | None]:
    host = host or ollama_host()
    prompt = _make_prompt(prompt_tokens)
    url = _api_url(host, "/api/generate")

    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {**options, "num_predict": gen_tokens},
    }).encode()

    try:
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None, None

    prompt_count = data.get("prompt_eval_count", 0)
    prompt_duration_ns = data.get("prompt_eval_duration", 0)
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)

    pp_tok_s: float | None = None
    tg_tok_s: float | None = None

    if prompt_count and prompt_duration_ns:
        pp_tok_s = prompt_count / (prompt_duration_ns / 1e9)
    if eval_count and eval_duration_ns:
        tg_tok_s = eval_count / (eval_duration_ns / 1e9)

    return pp_tok_s, tg_tok_s


def ollama_reproducible_command(model: str, params: BenchParams) -> str:
    opts = {
        "num_gpu": params.ngl,
        "num_batch": params.batch_size,
        "num_ctx": max(params.pp + params.tg, 512),
    }
    if params.threads > 0:
        opts["num_thread"] = params.threads
    return (
        f"ollama run {model} -- {json.dumps(opts)}\n"
        f"# See: {OLLAMA_DEFAULT_HOST}/api/tags"
    )


def ollama_runner(
    model: str,
    params: BenchParams | None = None,
    binary: str | None = None,
    timeout: int = 300,
    warmup: bool = True,
    **kwargs,
) -> TrialResult:
    params = params or BenchParams()

    options: dict[str, Any] = {
        "num_gpu": params.ngl,
        "num_batch": params.batch_size,
        "num_ctx": max(params.pp + params.tg, 512),
    }
    if params.threads > 0:
        options["num_thread"] = params.threads

    host = ollama_host()

    if warmup:
        warmup_opts = {**options, "num_gpu": min(params.ngl, 1), "num_predict": 4}
        warmup_body = json.dumps({
            "model": model,
            "prompt": "warmup",
            "stream": False,
            "options": warmup_opts,
        }).encode()
        try:
            warmup_req = urllib.request.Request(
                _api_url(host, "/api/generate"), data=warmup_body, method="POST"
            )
            warmup_req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(warmup_req, timeout=min(timeout, 60)):
                pass
        except (urllib.error.URLError, OSError):
            pass

    pp_tok_s, tg_tok_s = run_ollama_bench(
        model=model,
        options=options,
        prompt_tokens=params.pp,
        gen_tokens=params.tg,
        host=host,
        timeout=timeout,
    )

    return TrialResult(
        params=params,
        pp_tokens_s=pp_tok_s,
        tg_tokens_s=tg_tok_s,
        oom=(pp_tok_s is None and tg_tok_s is None),
        error=None if pp_tok_s is not None else "Ollama request failed",
    )
