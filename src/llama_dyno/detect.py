"""Hardware and backend detection for llama-dyno."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import warnings
from pathlib import Path

# Suppress pynvml deprecation warning from nvidia-ml-py
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")

from .types import HardwareFingerprint, IkFeatures

# Optional psutil for RAM detection
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def _is_wsl() -> bool:
    """Check if running under Windows Subsystem for Linux."""
    try:
        with open("/proc/version") as f:
            content = f.read().lower()
            return "microsoft" in content or "wsl" in content
    except Exception:
        return False


def _detect_gpu() -> tuple[str, int, str, str | None]:
    """Detect NVIDIA GPU info using pynvml. Returns (name, vram_mib, driver, cuda_version)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle).decode() if isinstance(
            pynvml.nvmlDeviceGetName(handle), bytes
        ) else pynvml.nvmlDeviceGetName(handle)
        vram = pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024 * 1024)
        driver = pynvml.nvmlDeviceGetDriverVersion(handle).decode() if isinstance(
            pynvml.nvmlDeviceGetDriverVersion(handle), bytes
        ) else pynvml.nvmlDeviceGetDriverVersion(handle)
        # CUDA version from NVML
        try:
            cuda_major = pynvml.nvmlSystemGetCudaDriverVersion_v2() // 1000
            cuda_minor = (pynvml.nvmlSystemGetCudaDriverVersion_v2() % 1000) // 10
            cuda_ver = f"{cuda_major}.{cuda_minor}"
        except Exception:
            cuda_ver = None
        pynvml.nvmlShutdown()
        return name, int(vram), driver, cuda_ver
    except Exception as e:
        # Fallback: try nvidia-smi
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True,
            )
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            name = parts[0]
            vram = int(parts[1])
            driver = parts[2]

            cuda_ver = None
            try:
                out2 = subprocess.run(
                    ["nvidia-smi"], capture_output=True, text=True, timeout=10
                )
                m = re.search(r"CUDA Version:\s*([\d.]+)", out2.stdout)
                if m:
                    cuda_ver = m.group(1)
            except Exception:
                pass
            return name, vram, driver, cuda_ver
        except Exception:
            return "Unknown", 0, "unknown", None


def _detect_cpu() -> tuple[str, int]:
    """Detect CPU info."""
    cpu_name = "Unknown"
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=5,
            )
            lines = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
            if len(lines) > 1:
                cpu_name = lines[1]
        elif platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu_name = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass
        elif platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            cpu_name = out.stdout.strip()
    except Exception:
        pass

    cores = os.cpu_count() or 0
    return cpu_name, cores


def _detect_ram() -> int:
    """Detect total RAM in MiB."""
    if HAS_PSUTIL:
        return int(psutil.virtual_memory().total // (1024 * 1024))
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        return int(parts[1]) // 1024  # kB -> MiB
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            return int(out.stdout.strip()) // (1024 * 1024)
        except Exception:
            pass
    elif os.name == "nt":
        try:
            out = subprocess.run(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=5,
            )
            lines = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
            if len(lines) > 1:
                return int(lines[1]) // (1024 * 1024)
        except Exception:
            pass
    return 0


def _find_binary(name: str) -> str | None:
    """Find a binary in PATH. Returns path or None."""
    # WSL detection: prefer Linux binaries; Windows binaries under /mnt/c/
    # would need a separate discovery step.
    if os.name == "nt":
        name_exe = f"{name}.exe"
    else:
        name_exe = name
    for p in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(p, name_exe)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _get_commit_from_binary(binary: str) -> str | None:
    """Try to extract git commit from a llama.cpp binary."""
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        # Common patterns: "commit: abc1234" or "build: abc1234"
        for line in (result.stdout + result.stderr).splitlines():
            m = re.search(r"(?:commit|build|version)[\s:]+([a-f0-9]{7,40})", line, re.IGNORECASE)
            if m:
                return m.group(1)
        # Also try just running with no args
        m = re.search(r"([a-f0-9]{7,40})", result.stdout + result.stderr)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def _get_commit_from_git(directory: str | None = None) -> str | None:
    """Try to get commit hash from a local git repo."""
    try:
        if directory:
            result = subprocess.run(
                ["git", "-C", directory, "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
        else:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def detect_ik_features(binary_path: str | None = None) -> IkFeatures:
    """Detect which ik_llama.cpp feature flags the binary supports.

    Probes the binary's help output to check for -fmoe, -rtr, -amb flags.
    Returns IkFeatures with detected=True only if the binary name includes "ik_".
    """
    from .bench import find_bench_binary

    if binary_path is None:
        binary_path = find_bench_binary()
    if binary_path is None:
        return IkFeatures(detected=False)

    binary_name = os.path.basename(binary_path)
    if "ik_" not in binary_name:
        return IkFeatures(detected=False)

    # Probe via -h to see which flags are mentioned in the help text
    try:
        result = subprocess.run(
            [binary_path, "-h"],
            capture_output=True, text=True, timeout=5,
        )
        help_text = result.stdout + result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return IkFeatures(detected=True)

    return IkFeatures(
        detected=True,
        fmoe="-fmoe" in help_text,
        rtr="-rtr" in help_text,
        amb="-amb" in help_text,
    )


def detect_hardware() -> HardwareFingerprint:
    """Fingerprint the current hardware and detect llama.cpp backend."""
    gpu_name, vram, driver, cuda_ver = _detect_gpu()
    cpu_name, cpu_cores = _detect_cpu()
    ram = _detect_ram()

    # Detect backend
    backend = "llama.cpp"
    commit = None

    llama_bench = _find_binary("llama-bench")
    if llama_bench:
        commit = _get_commit_from_binary(llama_bench)

    # Check for ik_llama.cpp binary
    ik_bench = _find_binary("ik_llama-bench") or _find_binary("ik_llama-cli")
    if ik_bench:
        backend = "ik_llama.cpp"
        if not commit:
            commit = _get_commit_from_binary(ik_bench)

    # If we couldn't extract from binary, try environment
    if not commit:
        for var in ["LLAMA_COMMIT", "IK_LLAMA_COMMIT"]:
            if var in os.environ:
                commit = os.environ[var]
                break

    # Detect ik_llama.cpp features if applicable
    ik_features = None
    if backend == "ik_llama.cpp" or (ik_bench and "ik_" in os.path.basename(ik_bench)):
        probe_binary = ik_bench or llama_bench
        if probe_binary:
            ik_features = detect_ik_features(probe_binary)

    return HardwareFingerprint(
        gpu_name=gpu_name,
        vram_total_mib=vram,
        driver_version=driver,
        cuda_version=cuda_ver,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        ram_total_mib=ram,
        backend=backend,
        backend_commit=commit,
        ik_features=ik_features,
    )
