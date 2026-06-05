# Contributing to Local TTS Studio

Thanks for your interest in improving Local TTS Studio! This project is a 100% local,
open-source speech studio, and contributions of all sizes are welcome — bug fixes, new
TTS engines, docs, and UI polish.

## Ways to contribute

- 🐛 **Report bugs** — open an issue with steps to reproduce, your OS, GPU, and Python version.
- 💡 **Suggest features** — open an issue describing the use case.
- 🔌 **Add a TTS engine** — see the provider interface in `infra/` (e.g. a Piper or Kokoro adapter).
- 📝 **Improve docs** — README, `docs/`, or inline docstrings.
- 🎨 **UI improvements** — the frontend is a single `simple-ui.html` file (no build step).

## Development setup

```bash
git clone https://github.com/sammy995/Local-TTS-Studio.git
cd Local-TTS-Studio

# Windows
./install.ps1

# macOS / Linux
./install.sh
```

Or manually:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python run_local.py
```

The app serves at **http://localhost:8000**.

## Project structure

This codebase follows a hexagonal (ports & adapters) layout — please keep the boundaries clean:

| Layer | Folder | Rule |
|-------|--------|------|
| Pure compute | `core/` | No I/O, no FastAPI, no environment branching. Returns plain types. |
| Orchestration | `services/` | Stateless. Storage/engines injected per call. |
| Side effects | `infra/` | Adapters (storage, future providers). Protocol-based. |
| Delivery | `runtimes/` | Thin FastAPI wrapper: parse request → call service → respond. |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full rules.

## Pull request process

1. **Fork** and create a branch — `git checkout -b feature/your-feature` or `fix/the-bug`.
2. **Keep changes focused** — one logical change per PR.
3. **Match the existing style** — follow the surrounding code; run `ruff check .` if you installed the dev extras.
4. **Smoke test** — make sure the app still imports and starts:
   ```bash
   python -c "import py_compile, glob; [py_compile.compile(f, doraise=True) for f in glob.glob('**/*.py', recursive=True)]"
   ```
5. **Open a PR** with a clear description of what changed and why. Link any related issue.

## Code style

- Python 3.10+, type hints where practical.
- Keep `core/` free of side effects and framework imports.
- Prefer composition and per-call dependency injection over global state.

## Reporting security issues

Please do **not** open public issues for security problems. See [`SECURITY.md`](SECURITY.md).

## Code of conduct

By participating, you agree to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).
