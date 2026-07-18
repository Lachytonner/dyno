"""Dyno — auto-tune and benchmark llama.cpp inference on NVIDIA GPUs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llama-dyno")
except PackageNotFoundError:  # not installed (e.g. running from a raw checkout)
    __version__ = "0.0.0+unknown"
