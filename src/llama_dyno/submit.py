"""Submit benchmark results to GitHub Gist."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from .report import format_json


def _gh_installed() -> bool:
    """Check if gh CLI is available."""
    try:
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, OSError):
        return False


def _gh_auth_status() -> bool:
    """Check if gh is authenticated."""
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, OSError):
        return False


def _gpu_key(hardware: dict[str, Any]) -> str:
    """Generate a filesystem-safe key for GPU model."""
    name = hardware.get("gpu_name", "unknown-gpu")
    # Clean for filename
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
    safe = safe.replace(" ", "_")
    return safe.lower()


def _model_key(model: dict[str, Any]) -> str:
    """Generate a filesystem-safe key for model name."""
    name = model.get("name", "unknown-model")
    if name.endswith(".gguf"):
        name = name[:-5]
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
    safe = safe.replace(" ", "_")
    return safe.lower()


def _result_filename(report: dict[str, Any]) -> str:
    """Generate the result filename: GPU_MODEL_quant_hash_vX.json."""
    hw = report.get("hardware", {})
    model = report.get("model", {})
    quant = model.get("quantization", "unknown")
    short_hash = model.get("sha256", "unknown")[:12] if model.get("sha256") else "unknown"
    version = report.get("dyno_version", "0.0.0")
    gpu = _gpu_key(hw)
    model_name = _model_key(model)
    return f"{gpu}_{model_name}_{quant}_{short_hash}_v{version}.json"


def _result_json_content(report: dict[str, Any]) -> str:
    """Format the report JSON for submission."""
    return format_json(report)


def submit_via_gist(report: dict[str, Any]) -> str | None:
    """Submit results as a GitHub Gist.

    Checks that gh is installed and authenticated before attempting.
    Returns the Gist URL on success or None on failure.
    """
    if not _gh_installed():
        print("Submit failed: gh not installed")
        return None

    if not _gh_auth_status():
        print("Submit failed: not authenticated (run 'gh auth login')")
        return None

    filename = _result_filename(report)
    content = _result_json_content(report)
    description = (
        f"Dyno benchmark: "
        f"{report.get('hardware', {}).get('gpu_name', 'GPU')}"
        f" + {report.get('model', {}).get('name', 'model')}"
    )

    try:
        result = subprocess.run(
            ["gh", "gist", "create", "--filename", filename,
             "--description", description, "-"],
            input=content,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

        print(f"Submit failed: unknown error — {result.stderr.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"Submit failed: unknown error — {e}")

    return None


def submit_report(report: dict[str, Any]) -> str:
    """Submit report, trying Gist first then saving locally.

    Returns the Gist URL on success.
    Raises RuntimeError if submission fails (with a message including the local path).
    """
    url = submit_via_gist(report)
    if url:
        return url

    # Gist failed — save locally
    result_dir = os.path.join(os.getcwd(), "dyno-results")
    os.makedirs(result_dir, exist_ok=True)
    filename = _result_filename(report)
    path = os.path.join(result_dir, filename)
    with open(path, "w") as f:
        f.write(_result_json_content(report))

    raise RuntimeError(
        "Could not submit via Gist. "
        "Install gh: gh auth login\n\n"
        f"Result saved locally to: {path}\n"
        "You can manually submit it at https://gist.github.com"
    )
