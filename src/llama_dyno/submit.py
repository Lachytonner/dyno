"""Submit benchmark results to GitHub PR or Gist fallback."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
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
    """Generate the result filename: GPU_MODEL_quant.json."""
    hw = report.get("hardware", {})
    model = report.get("model", {})
    quant = model.get("quantization", "unknown")
    return f"{_gpu_key(hw)}_{_model_key(model)}_{quant}.json"


def _result_json_content(report: dict[str, Any]) -> str:
    """Format the report JSON for submission."""
    return format_json(report)


def submit_via_pr(report: dict[str, Any]) -> str | None:
    """Submit results via a GitHub PR to llama-dyno-results repo.

    Uses gh CLI to fork, branch, commit, and open a PR.
    Returns the PR URL or None on failure.
    """
    if not _gh_installed():
        return None

    if not _gh_auth_status():
        return None

    target_repo = "lachy/llama-dyno-results"
    filename = _result_filename(report)
    content = _result_json_content(report)

    # Use temp dir for git operations
    with tempfile.TemporaryDirectory(prefix="dyno-submit-") as tmpdir:
        try:
            # Create the file in a subdirectory structure: results/GPU_MODEL_quant.json
            result_path = os.path.join(tmpdir, "results", filename)
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, "w") as f:
                f.write(content)

            # Determine branch name
            hw = report.get("hardware", {})
            model = report.get("model", {})
            branch = f"dyno/{_gpu_key(hw)}/{_model_key(model)}"

            # Use gh to create PR directly (gh pr create works without cloning)
            # Try gh pr create --fill with web mode first
            title = f"Add benchmark: {hw.get('gpu_name', 'Unknown GPU')} + {model.get('name', 'Unknown model')}"
            body = (
                f"## Dyno Benchmark Submission\n\n"
                f"**GPU:** {hw.get('gpu_name')}\n"
                f"**VRAM:** {hw.get('vram_total_mib')} MiB\n"
                f"**Model:** {model.get('name')}\n"
                f"**Backend:** {hw.get('backend')} ({hw.get('backend_commit', '')})\n"
                f"**TG throughput:** {report.get('results', {}).get('median_tg_tokens_per_sec', 'N/A')} tok/s\n\n"
                f"Auto-submitted by [llama-dyno](https://github.com/lachy/llama-dyno) v{report.get('dyno_version', '?')}"
            )

            # Try using gh to create PR from remote
            # gh doesn't support creating a branch on a remote we don't own,
            # so we'd need to fork. Let's use the API approach instead.

            # Check if the repo exists first
            check = subprocess.run(
                ["gh", "repo", "view", target_repo],
                capture_output=True, text=True, timeout=10,
            )
            if check.returncode != 0:
                # Try creating the results repo
                subprocess.run(
                    ["gh", "repo", "create", target_repo, "--public", "--description",
                     "Crowdsourced llama.cpp benchmarks from Dyno"],
                    capture_output=True, text=True, timeout=15,
                )

            # Use `gh pr create` which handles forking automatically
            # We need a local clone approach
            subprocess.run(
                ["gh", "repo", "fork", target_repo, "--clone", "--remote=true"],
                capture_output=True, cwd=tmpdir, timeout=30,
            )

            fork_dir = os.path.join(tmpdir, "llama-dyno-results")
            if not os.path.isdir(fork_dir):
                # Try alternate naming
                fork_dir = os.path.join(tmpdir, "results")
                if not os.path.isdir(fork_dir):
                    # Just use the tmpdir
                    fork_dir = tmpdir

            # Copy file into repo
            results_dir = os.path.join(fork_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, filename), "w") as f:
                f.write(content)

            # Commit and push
            subprocess.run(["git", "add", "."], cwd=fork_dir,
                           capture_output=True, timeout=10)
            subprocess.run(
                ["git", "commit", "-m", title],
                cwd=fork_dir, capture_output=True, timeout=10,
            )
            push = subprocess.run(
                ["git", "push", "origin", f"HEAD:{branch}"],
                cwd=fork_dir, capture_output=True, text=True, timeout=60,
            )

            if push.returncode != 0:
                return None

            # Create PR
            pr = subprocess.run(
                ["gh", "pr", "create",
                 "--repo", target_repo,
                 "--title", title,
                 "--body", body],
                cwd=fork_dir, capture_output=True, text=True, timeout=30,
            )

            if pr.returncode == 0 and pr.stdout.strip():
                return pr.stdout.strip()

        except (subprocess.TimeoutExpired, OSError, FileNotFoundError) as e:
            return None

    return None


def submit_via_gist(report: dict[str, Any]) -> str | None:
    """Submit results as a GitHub Gist (fallback).

    Returns the Gist URL or None on failure.
    """
    if not _gh_installed():
        return None

    filename = _result_filename(report)
    content = _result_json_content(report)

    try:
        # gh gist create uses stdin
        result = subprocess.run(
            ["gh", "gist", "create", "--filename", filename,
             "--description", f"Dyno benchmark: {report.get('hardware', {}).get('gpu_name', 'GPU')}",
             "-"],
            input=content,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return None


def submit_report(report: dict[str, Any]) -> str:
    """Submit report, trying PR first then Gist fallback.

    Returns the URL of the submission.
    """
    url = submit_via_pr(report)
    if url:
        return url

    url = submit_via_gist(report)
    if url:
        return url

    # If neither works, save locally and tell user
    import os
    result_dir = os.path.join(os.getcwd(), "dyno-results")
    os.makedirs(result_dir, exist_ok=True)
    filename = _result_filename(report)
    path = os.path.join(result_dir, filename)
    with open(path, "w") as f:
        f.write(_result_json_content(report))

    raise RuntimeError(
        "Could not submit via PR or Gist. "
        "Install and authenticate the GitHub CLI:\n"
        "  gh auth login\n\n"
        f"Result saved locally to: {path}\n"
        "You can manually submit it to https://github.com/lachy/llama-dyno-results"
    )
