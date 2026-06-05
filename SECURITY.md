# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability, please report it **privately** rather than opening a
public issue.

- Use GitHub's [private vulnerability reporting](https://github.com/sammy995/Local-TTS-Studio/security/advisories/new), or
- Open an issue titled "Security — contact request" (without details) and a maintainer will reach out for a private channel.

Please include:

- A description of the issue and its impact
- Steps to reproduce
- Affected version / commit
- Any suggested mitigation

We aim to acknowledge reports within a few days and to keep you updated as we investigate.

## Scope and threat model

Local TTS Studio is designed to run **locally** on a user's own machine and to bind to
`localhost` by default. It performs no telemetry and stores generated audio on the local disk.

Key things to keep in mind:

- The default config binds to `0.0.0.0:8000`. If you run it on a shared or untrusted network,
  restrict the host/port or place it behind a reverse proxy — the API is unauthenticated by design
  for local use.
- Optional music-library API keys (Jamendo, Freesound) and any future cloud provider keys live in
  `config.yaml` / `.env`, which are git-ignored. Never commit real keys.
- Voice cloning operates on audio you provide. Only clone voices you have the right to use.

## Supported versions

The latest release on the `main` branch receives security fixes. Older tagged releases are not
maintained.
