"""Data types for llama-dyno."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class HardwareFingerprint:
    gpu_name: str
    vram_total_mib: int
    driver_version: str
    cuda_version: str | None
    cpu_name: str
    cpu_cores: int
    ram_total_mib: int
    backend: str  # "llama.cpp" or "ik_llama.cpp"
    backend_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchParams:
    ngl: int = 99          # GPU layers
    flash_attn: bool = True
    ct_k: str = "f16"      # K cache quant
    ct_v: str = "f16"      # V cache quant
    batch_size: int = 512
    ubatch_size: int = 512
    threads: int = 0       # 0 = auto
    fmoe: bool = False     # ik_llama.cpp
    rtr: bool = False      # ik_llama.cpp
    amb: bool = False      # ik_llama.cpp

    pp: int = 512          # prompt tokens (fixed for bench)
    tg: int = 128          # generation tokens (fixed for bench)

    def to_flag_list(self) -> list[str]:
        flags = [
            "-ngl", str(self.ngl),
            "-b", str(self.batch_size),
            "-ub", str(self.ubatch_size),
            "-ctk", self.ct_k,
            "-ctv", self.ct_v,
            "-p", str(self.pp),
            "-n", str(self.tg),
        ]
        if self.threads > 0:
            flags.extend(["-t", str(self.threads)])
        if self.flash_attn:
            flags.append("--flash-attn")
            flags.append("on")
        else:
            flags.append("--flash-attn")
            flags.append("off")
        if self.fmoe:
            flags.append("-fmoe")
        if self.rtr:
            flags.append("-rtr")
        if self.amb:
            flags.append("-amb")
        return flags

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def clone(self) -> BenchParams:
        return BenchParams(**{k: v for k, v in asdict(self).items()
                             if k in BenchParams.__dataclass_fields__})


@dataclass
class TrialResult:
    """Result from a single llama-bench trial."""
    params: BenchParams
    pp_ms: float | None = None       # prompt processing time ms
    pp_token_ms: float | None = None # per-token pp time
    tg_ms: float | None = None       # text generation time ms
    tg_token_ms: float | None = None # per-token tg time
    pp_tokens_s: float | None = None # tokens/sec pp
    tg_tokens_s: float | None = None # tokens/sec tg
    oom: bool = False
    error: str | None = None

    @property
    def score(self) -> float:
        """Combined score weighting both pp and tg throughput."""
        if self.oom or self.error:
            return -1.0
        pp = self.pp_tokens_s or 0
        tg = self.tg_tokens_s or 0
        # Weight prompt processing less than generation (typically 1:1 or 1:2)
        return pp * 0.3 + tg * 0.7


@dataclass
class TuneResult:
    winning_params: BenchParams
    trials: list[TrialResult] = field(default_factory=list)
    median_pp_tokens_s: float | None = None
    median_tg_tokens_s: float | None = None
    variance_pp: float | None = None
    variance_tg: float | None = None


@dataclass
class ModelInfo:
    path: str
    name: str
    sha256: str
    quantization: str | None = None
    file_size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def model_info_from_path(path: str) -> ModelInfo:
    p = Path(path)
    info = ModelInfo(
        path=str(p.resolve()),
        name=p.name,
        sha256=compute_sha256(path),
        file_size_bytes=p.stat().st_size,
    )
    # Try to extract quantization from GGUF filename convention
    name = p.stem
    for q in ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1",
              "Q4_K_S", "Q4_K_M", "Q4_K_L", "Q5_0", "Q5_1",
              "Q5_K_S", "Q5_K_M", "Q5_K_L", "Q6_K", "Q8_0",
              "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
              "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ3_L",
              "IQ4_XS", "IQ4_NL", "IQ4_NL_XL",
              "BF16", "F16", "F32"]:
        if f"-{q}" in name or f"_{q}" in name:
            info.quantization = q
            break

    # Fallback: scan for -Q or _Q patterns followed by a number
    # (handles QAT-style quants like QAT-Q4_0 that may not be in the known list)
    if info.quantization is None:
        m = re.search(r'[-_]([qQ]\d[-_\w]*)', name)
        if m:
            info.quantization = m.group(1)

    return info


def default_report(
    model: ModelInfo,
    hardware: HardwareFingerprint,
    tune_result: TuneResult,
    dyno_version: str = "0.1.0",
) -> dict:
    return {
        "dyno_version": dyno_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware.to_dict(),
        "model": model.to_dict(),
        "winning_params": tune_result.winning_params.to_dict(),
        "results": {
            "median_pp_tokens_per_sec": tune_result.median_pp_tokens_s,
            "median_tg_tokens_per_sec": tune_result.median_tg_tokens_s,
            "variance_pp": tune_result.variance_pp,
            "variance_tg": tune_result.variance_tg,
        },
        "trial_count": len(tune_result.trials),
    }
