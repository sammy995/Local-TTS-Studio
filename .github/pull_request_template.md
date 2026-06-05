## What does this PR do?

<!-- A clear description of the change and why it's needed. -->

## Related issue

<!-- e.g. Closes #123 — or "none" -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] New TTS engine / provider
- [ ] Documentation
- [ ] Refactor / chore

## Checklist

- [ ] I followed the hexagonal layout — `core/` stays pure (see [ARCHITECTURE.md](../ARCHITECTURE.md))
- [ ] The app still imports and starts (`python run_local.py`)
- [ ] `python -m compileall core services infra runtimes run_local.py` passes
- [ ] I updated the README / docs if behavior changed
