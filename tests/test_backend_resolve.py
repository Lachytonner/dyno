"""Test backend auto-detection logic."""

from unittest.mock import patch

from llama_dyno.cli import _resolve_backend, _build_runner


def test_resolve_gguf_file(tmp_path):
    """A .gguf file on disk should return 'gguf'."""
    gguf = tmp_path / "test.gguf"
    gguf.write_text("fake gguf")
    backend, name = _resolve_backend(str(gguf))
    assert backend == "gguf"
    assert name == str(gguf)


def test_resolve_ollama_exact_match():
    """Exact model name match in Ollama list."""
    with patch("llama_dyno.cli.ollama_available", return_value=True), \
         patch("llama_dyno.cli.list_ollama_models", return_value=["llama3:latest", "mistral:7b"]):
        backend, name = _resolve_backend("llama3:latest")
        assert backend == "ollama"
        assert name == "llama3:latest"


def test_resolve_ollama_prefix_match():
    """Prefix match ('llama3' matches 'llama3:latest'). Returns resolved name."""
    with patch("llama_dyno.cli.ollama_available", return_value=True), \
         patch("llama_dyno.cli.list_ollama_models", return_value=["llama3:latest"]):
        backend, name = _resolve_backend("llama3")
        assert backend == "ollama"
        assert name == "llama3:latest"  # Bug fix: must return resolved, not original


def test_resolve_unknown():
    """Model not found anywhere returns 'unknown'."""
    with patch("llama_dyno.cli.ollama_available", return_value=False), \
         patch("llama_dyno.cli.lmstudio_available", return_value=False):
        backend, name = _resolve_backend("nonexistent-model-xyz")
        assert backend == "unknown"


def test_resolve_lmstudio_substring_match():
    """Substring match in LM Studio models (exact → prefix → substring)."""
    with patch("llama_dyno.cli.ollama_available", return_value=False), \
         patch("llama_dyno.cli.lmstudio_available", return_value=True), \
         patch("llama_dyno.cli.list_lmstudio_models", return_value=["meta-llama-3.2-3b-instruct"]):
        backend, name = _resolve_backend("llama-3.2-3b")
        assert backend == "lmstudio"
        assert name == "meta-llama-3.2-3b-instruct"


def test_prefix_match_runner_uses_resolved_name():
    """Prefix match resolves to exact name, and the runner is invoked with it."""
    with patch("llama_dyno.cli.ollama_available", return_value=True), \
         patch("llama_dyno.cli.list_ollama_models", return_value=["llama3:8b-instruct-q4_0", "other-model"]):
        backend, resolved = _resolve_backend("llama3")
        assert backend == "ollama"
        assert resolved == "llama3:8b-instruct-q4_0"

        runner, tune_kw = _build_runner(backend)
        assert runner is not None
        assert tune_kw == {"sweep_fa": False, "sweep_kv": False}

        # Verify the runner wrapper actually calls ollama_runner with the resolved name
        with patch("llama_dyno.cli.ollama_runner") as mock_runner:
            mock_runner.return_value = None  # TrialResult is built inside; return mock
            runner(resolved, timeout=10, warmup=False)
            mock_runner.assert_called_once()
            # First positional arg should be the resolved name
            assert mock_runner.call_args[0][0] == "llama3:8b-instruct-q4_0"
