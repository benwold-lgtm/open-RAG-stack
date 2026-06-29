# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security vulnerabilities.

Report privately through GitHub: on the repository's **Security** tab, click
**"Report a vulnerability"** (GitHub private vulnerability reporting). Include the
affected component, a description, and reproduction steps if you have them.

This is a community project maintained on a best-effort basis — there is **no
guaranteed response time or SLA**. You'll get a reply when the maintainer is able.

## Supported versions

Only the current `main` branch and the `:latest` container images built from it are
maintained. There are no long-term support branches and no security backports to older
tags — pin to a specific image digest if you need stability, and update to pick up fixes.

## Scope and shared responsibility

This is **self-hosted software**. The project ships sensible secure-by-default config
where it can (e.g. the data plane fails closed in `production`, see the README), but the
security of a deployment depends on how it is operated. Operators are responsible for:

- **Network exposure** — keep the NodePorts / host ports on a trusted, isolated network;
  the services are not hardened to be exposed directly to the internet.
- **TLS** — terminate TLS in front of the chat UI and set `cookieSecure: true`.
- **Secrets** — protect the Kubernetes Secrets / `.env` values; consider encryption at
  rest or an external secrets manager.
- **Authentication** — set a `SERVICE_TOKEN` to lock the data plane, and a strong
  `SESSION_SECRET` plus an auth method for the chat UI.

See **[Hardening for production use](README.md#hardening-for-production-use)** for the
full checklist. Misconfiguration of a self-hosted deployment is not a vulnerability in
this project, but doc gaps that lead operators astray are — please report those too.
