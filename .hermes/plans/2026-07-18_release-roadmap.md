# Dyno Release Roadmap

> Roadmap for improving Dyno from v0.1.0 toward v1.0.0.

---

## v0.2.0 — Reliability & Polish

**Goal:** Fix the rough edges that bite every user on first use.

- [ ] **Deduplicate trials** — don't re-bench the same config twice in the search loop (currently repeats identical params on the hill-climb re-check)
- [ ] **Fix QAT quant parsing** — detect non-standard quants like `QAT-Q4_0` (currently all show "unknown") by scanning for any `-Q` or `-q` pattern
- [ ] **Accept existing results** — `dyno report result.json` should work from a saved file instead of re-tuning from scratch
- [ ] **Handle long paths** — the "Next: dyno bench..." hint wraps and breaks in the terminal; truncate model path or use `...`
- [ ] **Better error messages** — when llama-bench exits non-zero but isn't OOM, show stderr clearly (currently shows raw stderr[:500])
- [ ] **Quiet mode / JSON-only output** — `dyno tune --json-out results.json` for scripting

**Verify:** All existing + 3 new tests. Test with the Gemma 12B QAT model to confirm quant parsing.

---

## v0.3.0 — Smarter Search

**Goal:** Faster, more accurate tuning with user control.

- [ ] **Configurable scoring** — let user set `--pp-weight 0.5 --tg-weight 0.5` or presets (`--optimize generation` / `--optimize prompt`)
- [ ] **Search pruning** — if ngl=50 beats ngl=99 by a wide margin, skip ngl=25 (don't waste trials on worse bets)
- [ ] **Per-trial stability** — show stddev across the 5 llama-bench samples, flag unstable configs
- [ ] **Warmup-aware** — run a short warmup before the measured run to stabilise GPU clocks (configurable via `--no-warmup`)
- [ ] **Adaptive trial budget** — stop early if the last N trials didn't improve the best score by more than 2%
- [ ] **Model metadata extraction** — read GGUF header for true param count, context length, n_layer to inform heuristics

**Verify:** Tune the same model twice with same mode and get nearly identical winners.

---

## v0.4.0 — ik_llama.cpp Support

**Goal:** Full support for ik_llama.cpp's extra features (fmoe, rtr, amb).

- [ ] **Detect ik features at runtime** — parse `--version` output or try a dry-run to check which flags the binary accepts
- [ ] **Validate fmoe/rtr/amb flags** — run a quick smoke test with each flag before including in the search space
- [ ] **ik-specific defaults** — enable fmoe by default for MoE models, set different thread heuristics
- [ ] **Feature-aware search space** — only try fmoe on/off when the model is MoE (detect from GGUF)
- [ ] **Document ik_llama.cpp install** — add brew tap / build instructions to README

**Verify:** Build ik_llama.cpp, run `dyno tune` on an MoE model, confirm fmoe trials appear in the progress table.

---

## v0.5.0 — Results & Community

**Goal:** Make `dyno submit` actually useful and build the results ecosystem.

- [ ] **Simplify submit** — drop the complex git dance; just `gh gist create` always (simpler, always works)
- [ ] **`dyno search <gpu> <model>`** — query the community results repo API (or a curated JSON index) for existing results matching your hardware
- [ ] **`dyno compare <file1.json> <file2.json>`** — side-by-side diff of two result files
- [ ] **Auto-version results** — include the GGUF file hash and dyno version in the filename so results never collide
- [ ] **Results registry** — a GitHub Pages site at `lachytonner.github.io/dyno-results` with a searchable table of all submissions

**Verify:** `dyno submit` creates a gist. `dyno search "RTX 4070"` returns the Gemma 4 result.

---

## v1.0.0 — Production

**Goal:** Battle-hardened, cross-platform, documented.

- [ ] **Windows + WSL 2 validation** — test and fix path handling, binary detection, nvidia-smi fallback on Windows
- [ ] **CI/CD** — GitHub Actions: run tests on Linux + Windows, lint with ruff, auto-publish to PyPI on tag
- [ ] **Structured logging** — add `--log-level debug` for troubleshooting
- [ ] **Full test coverage** — 80%+ line coverage: mock bench runner covers edge cases, CLI integration tests via Typer's CliRunner
- [ ] **`dyno doctor`** — check that llama-bench is installed, CUDA works, GPU is detected, model loads — single command for troubleshooting
- [ ] **Shell completions** — `dyno --install-completion`
- [ ] **Website** — single-page docs site with usage examples, all command references, FAQ

**Verify:** `pipx install llama-dyno` on a fresh Windows VM → `dyno detect` → `dyno tune` → `dyno report` all succeed. 100% of tests pass in CI on all three OSes.

---

## Stretch (post-1.0)

- Multi-GPU support (`--tensor-split`, `--split-mode`)
- AMD ROCm / Apple Metal backends
- Docker container with bundled llama.cpp + Dyno for one-shot `docker run`
- `dyno serve` — small web API that exposes tuning as a service
