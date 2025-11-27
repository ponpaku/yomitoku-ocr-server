# Repository Guidelines

This guide is for contributors working on YomiToku, a Japanese document AI engine. Keep changes small, tested, and well-described so reviews stay fast.

## Project Structure & Module Organization
- `src/yomitoku/`: core pipelines (OCR, layout parsing, table structure, exports); CLI entrypoints live in `src/yomitoku/cli/`.
- `app/`: FastAPI service (`app/main.py`) plus static assets for the demo UI.
- `configs/` and `schemas/`: model presets and JSON schema definitions used by the CLI and docs.
- `tests/`: pytest suites; fixtures and sample files sit in `tests/data/` and `tests/yaml/`.
- `demo/` and `static/`: example scripts and sample inputs/outputs for local verification; avoid committing new large artifacts here.
- `docs/` and `mkdocs.yml`: MkDocs site; `scripts/` and `macros/` generate schema-related content.

## Build, Test, and Development Commands
- Install runtime only: `pip install -e .`
- Install with tooling: `pip install -e .[mcp] pytest ruff mkdocs-material` (or `uv sync --dev` if you use the bundled `uv.lock`).
- Run CLI locally: `yomitoku path/to/image_or_dir -f md -o results -v --figure`
- Serve the API: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
- Tests: `pytest tests -q`
- Lint: `ruff check src tests`

## Coding Style & Naming Conventions
- Target Python 3.10–3.12; use 4-space indentation and type hints on public functions.
- Prefer functional units in `src/yomitoku/...` over ad-hoc scripts; keep CLI options mirrored in both `yomitoku/cli/main.py` and docs.
- Naming: snake_case for modules/functions/variables, PascalCase for classes, and kebab-case for CLI flags. Keep config keys consistent with existing YAML presets.
- Use `yomitoku.utils.logger` for logging and keep exports deterministic (no random filenames without seeds).

## Testing Guidelines
- Write pytest cases near related modules and reuse fixtures from `tests/data/` and `tests/yaml/`.
- Keep tests GPU-optional; prefer CPU-friendly paths (`-d cpu` / `--lite`) and small sample files.
- When adding new output formats or schemas, include golden-file comparisons or schema validation where practical.

## Commit & Pull Request Guidelines
- Commit messages in history are short and imperative (“fix”, “revise document”); follow that style and keep summaries under ~72 chars.
- PRs should describe the change, list key commands run (tests/lint), and link issues. Add screenshots or sample outputs when touching OCR/layout rendering or docs.
- Avoid committing model weights or generated outputs under `static/out/`; keep diffs focused and runnable with the documented commands.
