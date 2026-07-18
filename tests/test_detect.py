"""Tests for hardware detection, focused on the Apple Silicon path (mocked)."""

from __future__ import annotations

import subprocess

from llama_dyno import detect


def _fake_sysctl(out: str):
    def run(cmd, capture_output=True, text=True, timeout=5, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    return run


def test_apple_silicon_gpu_on_arm_mac(monkeypatch):
    monkeypatch.setattr(detect.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(detect.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(detect, "_detect_ram", lambda: 16384)
    monkeypatch.setattr(detect.subprocess, "run", _fake_sysctl("Apple M2 Max\n"))

    assert detect._apple_silicon_gpu() == ("Apple M2 Max", 16384)


def test_apple_silicon_gpu_none_on_linux(monkeypatch):
    monkeypatch.setattr(detect.platform, "system", lambda: "Linux")
    monkeypatch.setattr(detect.platform, "machine", lambda: "x86_64")
    assert detect._apple_silicon_gpu() is None


def test_apple_silicon_gpu_none_on_intel_mac(monkeypatch):
    monkeypatch.setattr(detect.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(detect.platform, "machine", lambda: "x86_64")
    assert detect._apple_silicon_gpu() is None


def test_detect_hardware_uses_apple_fallback(monkeypatch):
    monkeypatch.setattr(detect, "_detect_gpu", lambda: ("Unknown", 0, "unknown", None))
    monkeypatch.setattr(detect, "_apple_silicon_gpu", lambda: ("Apple M2 Max", 32768))
    monkeypatch.setattr(detect.platform, "mac_ver", lambda: ("14.5", ("", "", ""), "arm64"))
    monkeypatch.setattr(detect, "_find_binary", lambda name: None)

    hw = detect.detect_hardware()
    assert hw.gpu_name == "Apple M2 Max"
    assert hw.vram_total_mib == 32768
    assert hw.cuda_version is None
    assert hw.driver_version.startswith("macOS")


def test_detect_hardware_keeps_nvidia(monkeypatch):
    monkeypatch.setattr(detect, "_detect_gpu", lambda: ("NVIDIA RTX 4070", 12282, "550.0", "12.4"))

    def _boom():
        raise AssertionError("Apple fallback must not run when NVIDIA is present")

    monkeypatch.setattr(detect, "_apple_silicon_gpu", _boom)
    monkeypatch.setattr(detect, "_find_binary", lambda name: None)

    hw = detect.detect_hardware()
    assert hw.gpu_name == "NVIDIA RTX 4070"
    assert hw.vram_total_mib == 12282
    assert hw.cuda_version == "12.4"


def _fake_rocm_smi(stdout_map: dict[str, str]):
    """Build a subprocess.run replacement that returns stubbed rocm-smi JSON."""
    def run(cmd, capture_output=True, text=True, timeout=10, **kw):
        key = " ".join(cmd)
        out = stdout_map.get(key, "")
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    return run


ROCM_PRODUCT_JSON = '{"card0": {"Card series": "AMD Radeon RX 7900 XTX"}}'
ROCM_MEMINFO_JSON = '{"card0": {"VRAM Total Memory (B)": "25769803776"}}'


def test_amd_gpu_found(monkeypatch):
    monkeypatch.setattr(detect, "_find_binary", lambda name: "/opt/rocm/bin/rocm-smi" if name == "rocm-smi" else None)
    monkeypatch.setattr(detect.subprocess, "run", _fake_rocm_smi({
        "rocm-smi --showproductname --json": ROCM_PRODUCT_JSON,
        "rocm-smi --showmeminfo vram --json": ROCM_MEMINFO_JSON,
    }))

    result = detect._amd_gpu()
    assert result is not None
    name, vram = result
    assert "Radeon" in name
    assert vram == 24576  # 25769803776 // (1024*1024)


def test_amd_gpu_none_when_binary_missing(monkeypatch):
    monkeypatch.setattr(detect, "_find_binary", lambda name: None)
    assert detect._amd_gpu() is None


def test_amd_gpu_none_when_malformed_json(monkeypatch):
    monkeypatch.setattr(detect, "_find_binary", lambda name: "/opt/rocm/bin/rocm-smi" if name == "rocm-smi" else None)
    monkeypatch.setattr(detect.subprocess, "run", _fake_rocm_smi({
        "rocm-smi --showproductname --json": "not valid json",
        "rocm-smi --showmeminfo vram --json": "not valid json",
    }))

    assert detect._amd_gpu() is None


def test_detect_hardware_uses_amd_fallback(monkeypatch):
    monkeypatch.setattr(detect, "_detect_gpu", lambda: ("Unknown", 0, "unknown", None))
    monkeypatch.setattr(detect, "_amd_gpu", lambda: ("AMD Radeon RX 7900 XTX", 24576))
    monkeypatch.setattr(detect, "_find_binary", lambda name: None)

    hw = detect.detect_hardware()
    assert "Radeon" in hw.gpu_name
    assert hw.vram_total_mib == 24576
    assert hw.cuda_version is None
    assert hw.driver_version == "ROCm"


def test_detect_hardware_amd_before_apple(monkeypatch):
    monkeypatch.setattr(detect, "_detect_gpu", lambda: ("Unknown", 0, "unknown", None))
    monkeypatch.setattr(detect, "_amd_gpu", lambda: ("AMD Radeon RX 7900 XTX", 24576))

    def _boom():
        raise AssertionError("Apple fallback must not run when AMD is present")

    monkeypatch.setattr(detect, "_apple_silicon_gpu", _boom)
    monkeypatch.setattr(detect, "_find_binary", lambda name: None)

    hw = detect.detect_hardware()
    assert "Radeon" in hw.gpu_name
    assert hw.vram_total_mib == 24576


def test_detect_hardware_amd_not_consulted_when_nvidia_present(monkeypatch):
    monkeypatch.setattr(detect, "_detect_gpu", lambda: ("NVIDIA RTX 4070", 12282, "550.0", "12.4"))

    def _boom():
        raise AssertionError("AMD fallback must not run when NVIDIA is present")

    monkeypatch.setattr(detect, "_amd_gpu", _boom)
    monkeypatch.setattr(detect, "_find_binary", lambda name: None)

    hw = detect.detect_hardware()
    assert hw.gpu_name == "NVIDIA RTX 4070"
    assert hw.vram_total_mib == 12282
