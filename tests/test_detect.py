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
