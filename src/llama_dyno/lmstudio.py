"""LM Studio runner — benchmark LM Studio models through its OpenAI-compatible API.

Reuses the Dyno search/scoring/report machinery by satisfying the
run_bench signature so it slots into the existing injection seam.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .types import BenchParams, TrialResult

LMSTUDIO_DEFAULT_HOST = "http://localhost:1234/v1"


def lmstudio_base(default: str = LMSTUDIO_DEFAULT_HOST) -> str:
    return os.environ.get("LMSTUDIO_HOST", default)


def _api_url(host: str, path: str) -> str:
    base = host.rstrip("/")
    return f"{base}{path}"


def lmstudio_available(base: str | None = None) -> bool:
    base = base or lmstudio_base()
    try:
        req = urllib.request.Request(_api_url(base, "/models"), method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def list_lmstudio_models(base: str | None = None) -> list[str]:
    base = base or lmstudio_base()
    req = urllib.request.Request(_api_url(base, "/models"), method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode())
    return [m["id"] for m in data.get("data", [])]


def _make_prompt(num_tokens: int) -> str:
    words = ["hello", "world", "test", "prompt", "dyno", "benchmark"]
    repeat = max(1, num_tokens // len(words))
    return " ".join(words * repeat)


def _parse_sse_line(line: str) -> dict | None:
    if line.startswith("data: "):
        payload = line[6:].strip()
        if payload == "[DONE]":
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    return None


def run_lmstudio_bench(
    model: str,
    prompt_tokens: int = 512,
    gen_tokens: int = 128,
    base: str | None = None,
    timeout: int = 300,
) -> tuple[float | None, float | None]:
    base = base or lmstudio_base()
    prompt = _make_prompt(prompt_tokens)
    url = _api_url(base, "/chat/completions")

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "stream": True,
    }).encode()

    try:
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "text/event-stream")

        t0 = time.monotonic()
        first_token_time: float | None = None
        last_token_time: float | None = None
        completion_tokens = 0

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                line = resp.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue
                parsed = _parse_sse_line(line_str)
                if parsed is None:
                    continue
                choices = parsed.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        last_token_time = time.monotonic()
                        completion_tokens += 1
                usage = parsed.get("usage")
                if usage and "completion_tokens" in usage:
                    completion_tokens = usage["completion_tokens"]

        if first_token_time is None:
            return None, None

        ttft = first_token_time - t0
        gen_window = last_token_time - first_token_time if last_token_time and last_token_time > first_token_time else 0.001

        pp_tok_s: float | None = prompt_tokens / ttft if ttft > 0 else None
        tg_tok_s: float | None = completion_tokens / gen_window if gen_window > 0 else None

        return pp_tok_s, tg_tok_s

    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None, None


def lmstudio_reproducible_command(model: str, params: BenchParams) -> str:
    return (
        f"LM Studio: loaded model '{model}' via lms load\n"
        f"# Endpoint: {LMSTUDIO_DEFAULT_HOST}/chat/completions\n"
        f"# prompt_tokens={params.pp}, gen_tokens={params.tg}"
    )


def lmstudio_runner(
    model: str,
    params: BenchParams | None = None,
    binary: str | None = None,
    timeout: int = 300,
    warmup: bool = True,
    **kwargs,
) -> TrialResult:
    params = params or BenchParams()
    base = lmstudio_base()

    if warmup:
        warmup_body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "warmup"}],
            "max_tokens": 4,
            "stream": True,
        }).encode()
        try:
            warmup_req = urllib.request.Request(
                _api_url(base, "/chat/completions"), data=warmup_body, method="POST"
            )
            warmup_req.add_header("Content-Type", "application/json")
            warmup_req.add_header("Accept", "text/event-stream")
            with urllib.request.urlopen(warmup_req, timeout=min(timeout, 60)):
                pass
        except (urllib.error.URLError, OSError):
            pass

    pp_tok_s, tg_tok_s = run_lmstudio_bench(
        model=model,
        prompt_tokens=params.pp,
        gen_tokens=params.tg,
        base=base,
        timeout=timeout,
    )

    return TrialResult(
        params=params,
        pp_tokens_s=pp_tok_s,
        tg_tokens_s=tg_tok_s,
        oom=(pp_tok_s is None and tg_tok_s is None),
        error=None if pp_tok_s is not None else "LM Studio request failed",
    )
