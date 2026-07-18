"""Submit benchmark results: PR to the community repo, else Gist, else local."""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

from .report import format_json

# Community results repo that `dyno submit` opens PRs against.
COMMUNITY_REPO = "Lachytonner/llama-dyno-results"


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


def _run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess with captured output and a generous timeout."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)


def submit_via_pr(report: dict[str, Any]) -> str | None:
    """Open a PR against the community results repo with this benchmark.

    Forks + clones COMMUNITY_REPO, adds the result JSON under
    results/<gpu>/<filename>, pushes to the fork, and opens a PR via gh.
    Returns the PR URL, or None if gh is unavailable or any step fails (the
    caller then falls back to a Gist, then a local save).
    """
    if not _gh_installed() or not _gh_auth_status():
        return None

    filename = _result_filename(report)
    gpu = _gpu_key(report.get("hardware", {}))
    short_hash = (report.get("model", {}).get("sha256") or "result")[:12]
    branch = f"dyno-{gpu}-{short_hash}"
    title = (
        f"Add result: {report.get('hardware', {}).get('gpu_name', 'GPU')}"
        f" + {report.get('model', {}).get('name', 'model')}"
    )
    content = _result_json_content(report)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            if _run(["gh", "repo", "fork", COMMUNITY_REPO, "--clone"], cwd=tmp).returncode != 0:
                return None
            repo_dir = os.path.join(tmp, COMMUNITY_REPO.split("/")[-1])
            if not os.path.isdir(repo_dir):
                return None

            if _run(["git", "checkout", "-b", branch], cwd=repo_dir).returncode != 0:
                return None

            rel = f"results/{gpu}/{filename}"
            dest = os.path.join(repo_dir, "results", gpu, filename)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w") as f:
                f.write(content)

            for step in (["git", "add", rel], ["git", "commit", "-m", title],
                         ["git", "push", "-u", "origin", branch]):
                if _run(step, cwd=repo_dir).returncode != 0:
                    return None

            # Disambiguate the head ref for a cross-fork PR.
            who = _run(["gh", "api", "user", "-q", ".login"])
            head = f"{who.stdout.strip()}:{branch}" if who.returncode == 0 and who.stdout.strip() else branch

            pr = _run([
                "gh", "pr", "create", "--repo", COMMUNITY_REPO, "--head", head,
                "--title", title, "--body", "Automated benchmark submission via `dyno submit`.",
            ], cwd=repo_dir)
            if pr.returncode == 0 and pr.stdout.strip():
                return pr.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    return None


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
    """Submit report: PR to the community repo, else Gist, else save locally.

    Returns the PR or Gist URL on success.
    Raises RuntimeError if all remote paths fail (message includes the local path).
    """
    url = submit_via_pr(report)
    if url:
        return url

    url = submit_via_gist(report)
    if url:
        return url

    # Both remote paths failed — save locally
    result_dir = os.path.join(os.getcwd(), "dyno-results")
    os.makedirs(result_dir, exist_ok=True)
    filename = _result_filename(report)
    path = os.path.join(result_dir, filename)
    with open(path, "w") as f:
        f.write(_result_json_content(report))

    raise RuntimeError(
        "Could not submit via PR or Gist (is gh installed and authenticated? "
        "run 'gh auth login').\n\n"
        f"Result saved locally to: {path}\n"
        "You can open a PR manually at https://github.com/" + COMMUNITY_REPO
    )
