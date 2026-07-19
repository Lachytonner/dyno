# 🏎️ Dyno — llama.cpp Auto-Tuner & Benchmark

**Dyno** is an open-source CLI that auto-tunes and benchmarks LLM inference on NVIDIA, AMD (ROCm), and Apple Silicon GPUs — works with Ollama, LM Studio, and llama.cpp. Produces reproducible, shareable results.

```bash
pipx install llama-dyno

dyno detect
dyno tune ~/models/mistral-7b.Q4_K_M.gguf --quick
dyno bench ~/models/mistral-7b.Q4_K_M.gguf --ngl 99 --fa
dyno bench --lmstudio llama-3.2-3b  # benchmark an LM Studio model
dyno report ~/models/mistral-7b.Q4_K_M.gguf
dyno submit ~/models/mistral-7b.Q4_K_M.gguf
```

## 30-Second Quickstart

```bash
# 1. Install
pipx install llama-dyno

# 2. See your hardware and available models
dyno detect

# 3a. Optimize an Ollama model (auto-detected — no flags needed)
dyno optimize llama3

# 3b. Optimize an LM Studio model (auto-detected — no flags needed)
dyno optimize llama-3.2-3b

# 3c. Or optimize a GGUF file on disk
dyno optimize ~/Downloads/my-model.q4_k_m.gguf
```

## Prerequisites

- **Python 3.11+**
- **A GPU**: NVIDIA (drivers + CUDA), AMD (ROCm, via rocm-smi), or Apple Silicon (M-series, uses Metal + unified memory)
- **llama-bench** binary from [llama.cpp](https://github.com/ggml-org/llama.cpp) or [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) — built with the matching backend (CUDA or Metal)

Install llama.cpp:

```bash
brew install llama.cpp                        # macOS / Linux
# or build from source:
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release
```

ik_llama.cpp (optional, for MoE models):

```bash
git clone https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp && cmake -B build && cmake --build build --config Release
# Symlink the ik binaries
ln -sf $(pwd)/build/bin/ik_llama-bench ~/.local/bin/
```

## Commands

### `dyno detect`
Fingerprint hardware: GPU model, VRAM, driver/CUDA version, CPU, RAM, and detect which llama.cpp backend is installed (with commit hash).

### `dyno tune <model.gguf>`
Find the fastest config for your GPU + model.

| Flag | Default | Description |
|------|---------|-------------|
| `--quick` | (default) | ~10 trials, fast iteration |
| `--thorough` | | ~25 trials, best accuracy |

**Search strategy:**
1. **Validate** model loads correctly
2. **Coarse sweep** — ngl (layer offload), flash attention on/off, KV cache quant (f16 / q8_0 / q4_0)
3. **Hill-climb** — batch size (128–4096), threads (auto / half / full cores)
4. **ik_llama.cpp extras** — -fmoe, -rtr, -amb toggles (thorough only)

OOM configs are discarded gracefully with ngl backoff. Live progress table shows every trial.

## ik_llama.cpp Support

Dyno automatically detects [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) 
when `ik_llama-bench` is on your PATH. Extra flags tuned:

| Flag | Description | Best for |
|------|-------------|---------|
| `-fmoe` | Fast MoE computation | MoE models (Mixtral, DeepSeek) |
| `-rtr` | Runtime tensor reorder | Memory-bandwidth-bound scenarios |
| `-amb` | Attention memory bound | Large context / many KV heads |

Dyno detects which flags your build supports and only searches those.
MoE models get fmoe enabled by default; Dense models skip MoE flags entirely.

## LM Studio Support

Dyno can benchmark models served by [LM Studio](https://lmstudio.ai/) through its
OpenAI-compatible API (`http://localhost:1234/v1`).

```bash
# Show LM Studio status in hardware detection
dyno detect

# Benchmark a loaded model
dyno bench --lmstudio llama-3.2-3b

# Tune and create a report
dyno tune --lmstudio llama-3.2-3b --json-out report.json
```

LM Studio does not expose engine-reported tok/s — throughput is measured
**client-side** from the streamed SSE response (wall-clock). Results are
approximate and may vary with system load.

| Flag | Default | Description |
|------|---------|-------------|
| `--lmstudio` | | Benchmark an LM Studio model |
| `--host` | `http://localhost:1234/v1` | LM Studio API base URL |

### `dyno bench <model.gguf>`
Run a specific config 3× and report median tokens/sec with variance.

| Flag | Default | Description |
|------|---------|-------------|
| `--ngl` | 99 | GPU layers to offload |
| `--fa/--no-fa` | true | Flash attention |
| `--ctk` | f16 | K cache quant (f16, q8_0, q4_0) |
| `--ctv` | f16 | V cache quant (f16, q8_0, q4_0) |
| `--batch` | 512 | Batch size |
| `--ubatch` | 512 | Micro batch size |
| `--threads` | 0 (auto) | Thread count |
| `--runs` | 3 | Number of benchmark runs |
| `--fmoe` | false | Fast MoE (ik_llama.cpp) |
| `--rtr` | false | Runtime reorder (ik_llama.cpp) |
| `--amb` | false | Attention memory bound (ik_llama.cpp) |

### `dyno report <model.gguf>`
Generate a shareable report including:
- Full hardware fingerprint
- Model details (name, quant, SHA-256)
- Winning config
- Median scores with variance
- **Reproducible llama-server command**
- **Shareable markdown snippet**
- JSON output

### `dyno submit <model.gguf>`
Submit your results to the community results repo (`llama-dyno-results`):
1. Opens a PR via `gh` CLI
2. Falls back to a GitHub Gist
3. Saves locally if neither works

## Quality & Reproducibility

- **Fixed bench params:** Every run uses `pp=512` / `tg=128` (fixed prompt/gen tokens) so results are comparable
- **3-run median + variance** reported
- **No fabricated numbers** — real hardware detection, real subprocess results
- **OOM handling** — graceful backoff of ngl
- **Clear install hints** — suggests brew or git clone when binaries missing

## Results Table (Example)

| GPU | Model | Quant | Backend | TG tok/s |
|-----|-------|-------|---------|----------|
| RTX 4090 | Mixtral-8x7B | Q4_K_M | ik_llama.cpp | 58.7 |
| RTX 4090 | Llama-3-70B | Q4_K_M | ik_llama.cpp | 42.3 |
| RTX 4090 | Llama-3-70B | Q4_K_M | llama.cpp | 35.1 |
| RTX 3090 | Mistral-7B | Q4_K_M | llama.cpp | 112.8 |
| RTX 4060 | Phi-3-mini | Q4_K_M | llama.cpp | 68.5 |

## Shell Completions

Dyno supports shell completions via Typer/Click:

```bash
# Install completions for your shell
dyno --install-completion

# Show completion script (to manually install)
dyno --show-completion
```

Supported shells: bash, zsh, fish, powershell.

## Architecture

```
src/llama_dyno/
├── __init__.py    # Package metadata
├── cli.py         # Typer CLI (detect, tune, bench, report, submit)
├── types.py       # Data types (BenchParams, TrialResult, etc.)
├── detect.py      # Hardware fingerprinting (pynvml, nvidia-smi, /proc)
├── bench.py       # llama-bench subprocess driver + parser
├── tune.py        # Search strategy (coarse sweep + hill climb)
├── ollama.py      # Ollama REST API runner
├── lmstudio.py    # LM Studio OpenAI-compatible API runner
├── report.py      # Report generation (JSON + markdown)
└── submit.py      # GitHub PR / Gist submission
```

## Development

```bash
git clone https://github.com/lachy/llama-dyno
cd llama-dyno
pip install -e ".[dev]"
pytest
```

## Out of Scope (v1)

- GUI
- Multi-GPU
- Intel backends (planned)
- Server hosting for results

## License

Apache 2.0
