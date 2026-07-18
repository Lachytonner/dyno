"""Hardware and backend detection for llama-dyno."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import warnings

from .types import HardwareFingerprint, IkFeatures

# Suppress pynvml deprecation warning from nvidia-ml-py
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")

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
    except Exception:
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
            lines = [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
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
            lines = [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
            if len(lines) > 1:
                return int(lines[1]) // (1024 * 1024)
        except Exception:
            pass
    return 0


def _apple_silicon_gpu() -> tuple[str, int] | None:
    """Detect an Apple Silicon GPU as (name, usable memory MiB), else None.

    Apple GPUs share unified memory with the CPU — there is no discrete VRAM, so
    we report total RAM as the pool the GPU can draw from (macOS lets Metal use
    most of it).
    # ponytail: unified mem ≈ total RAM; refine via Metal recommendedMaxWorkingSetSize if OOM heuristics misfire
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    chip = ""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        chip = out.stdout.strip()
    except Exception:
        pass
    return chip or "Apple Silicon GPU", _detect_ram()


def _amd_gpu() -> tuple[str, int] | None:
    """Detect an AMD GPU via rocm-smi as (name, vram MiB), else None.
    # ponytail: rocm-smi JSON field names vary by version; parse defensively
    """
    if _find_binary("rocm-smi") is None:
        return None

    name = "Unknown AMD GPU"
    vram_mib = 0

    # Get product name from JSON (field names differ across ROCm versions)
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            data = json.loads(out.stdout)
            if isinstance(data, dict):
                for card_info in data.values():
                    if isinstance(card_info, dict):
                        name = (
                            card_info.get("Card series")
                            or card_info.get("Card model")
                            or card_info.get("Card SKU")
                            or name
                        )
                        break
    except Exception:
        pass

    if name == "Unknown AMD GPU":
        try:
            out = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0:
                for line in out.stdout.strip().splitlines():
                    if ":" in line:
                        parts = line.split(":", 1)
                        candidate = parts[1].strip()
                        if candidate:
                            name = candidate
                            break
        except Exception:
            pass

    # Get VRAM from rocm-smi --showmeminfo vram --json (bytes)
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            data = json.loads(out.stdout)
            if isinstance(data, dict):
                for card_info in data.values():
                    if isinstance(card_info, dict):
                        raw = card_info.get("VRAM Total Memory (B)")
                        if raw is not None:
                            vram_mib = int(raw) // (1024 * 1024)
                            break
    except Exception:
        pass

    if vram_mib == 0:
        return None

    return name, vram_mib


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
    # No NVIDIA GPU found — fall back to AMD, then Apple Silicon.
    if gpu_name == "Unknown" or vram == 0:
        amd = _amd_gpu()
        if amd:
            gpu_name, vram = amd
            driver = "ROCm"
            cuda_ver = None
        else:
            apple = _apple_silicon_gpu()
            if apple:
                gpu_name, vram = apple
                driver = f"macOS {platform.mac_ver()[0]}"
                cuda_ver = None
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
