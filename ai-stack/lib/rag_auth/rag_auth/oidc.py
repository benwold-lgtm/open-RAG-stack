"""OIDC JWT validation + the composite authenticator (the security-critical part).

Validate access tokens against the IdP's published keys (JWKS) with an **asymmetric-only**
algorithm allow-list, full claim checks (``iss``/``aud``/``exp``), and a bounded,
rate-limited key cache. The **service** owns ``group -> role -> scopes``; the IdP only asserts
group membership.

The :class:`CompositeAuthenticator` chains **OIDC -> static key -> 401**: an OIDC failure
(bad/expired token, IdP down) falls through to the static/break-glass keys, so OIDC fails
*closed* while break-glass keeps working through an IdP outage.

Threat-model notes carried from the blueprint:
* Never trust a key embedded in the token — only keys from the issuer's JWKS, matched by ``kid``.
* Reject ``alg: none`` and any ``HS*`` (symmetric) algorithm at config time.
* The issuer / JWKS URL should be run through an egress (SSRF) policy by the deployment;
  this module only fetches over the configured ``jwks_uri`` (or the issuer's discovery doc).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Optional, Protocol

import jwt

from .authenticator import Authenticator
from .errors import AuthConfigError, OIDCError
from .rbac import AUTH_OIDC, Principal, RoleScopes

# Asymmetric algorithms only. Symmetric (HS*) and 'none' are refused — they enable the
# classic algorithm-confusion / unsigned-token forgeries.
ASYMMETRIC_ALGS = frozenset(
    {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "PS256", "PS384", "PS512"}
)


@dataclass(frozen=True)
class OIDCConfig:
    issuer: str
    audience: str
    jwks_uri: Optional[str] = None  # discovered from issuer if not set (set explicitly when air-gapped)
    groups_claim: str = "groups"
    subject_claim: str = "sub"
    algorithms: frozenset[str] = frozenset({"RS256"})
    leeway: int = 60  # clock-skew seconds
    jwks_cache_ttl: int = 600
    jwks_min_refresh_interval: int = 30  # rate-limit unknown-kid refetch
    group_roles: Mapping[str, str] = field(default_factory=dict)  # IdP group -> role name
    default_role: Optional[str] = None  # role for an authenticated user with no mapped group

    def __post_init__(self) -> None:
        bad = self.algorithms - ASYMMETRIC_ALGS
        if bad or not self.algorithms:
            raise AuthConfigError(
                f"OIDC algorithms must be a non-empty subset of asymmetric algs; "
                f"refused: {sorted(bad) or 'empty'}"
            )


class JWKSResolver(Protocol):
    async def get_jwk(self, kid: str) -> dict:
        ...


class JWKSCache:
    """Bounded-TTL JWKS cache with rate-limited unknown-kid refetch.

    * Serves a known ``kid`` from cache while fresh (within ``ttl``).
    * On an unknown or stale ``kid``, refetches — but at most once per ``min_refresh_interval``
      (an unknown-kid flood must not become a fetch flood against the IdP).
    * On a fetch failure (IdP unreachable), serves a cached key if it has one — that's what
      makes OIDC fail *closed* but break-glass keep working.

    ``fetch`` is an async callable returning the JWKS ``keys`` array; ``clock`` is injectable
    for testing (defaults to a monotonic clock).
    """

    def __init__(
        self,
        fetch: Callable[[], Awaitable[list[dict]]],
        *,
        ttl: int = 600,
        min_refresh_interval: int = 30,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._fetch = fetch
        self._ttl = ttl
        self._min_refresh = min_refresh_interval
        self._clock = clock
        self._keys: dict[str, dict] = {}
        self._fetched_at: Optional[float] = None
        self._last_attempt: Optional[float] = None
        self._lock = asyncio.Lock()

    def _fresh(self, now: float) -> bool:
        return self._fetched_at is not None and (now - self._fetched_at) < self._ttl

    async def get_jwk(self, kid: str) -> dict:
        now = self._clock()
        if kid in self._keys and self._fresh(now):
            return self._keys[kid]

        async with self._lock:
            now = self._clock()
            # Re-check after acquiring the lock (another coroutine may have refreshed).
            if kid in self._keys and self._fresh(now):
                return self._keys[kid]

            rate_limited = (
                self._last_attempt is not None and (now - self._last_attempt) < self._min_refresh
            )
            if rate_limited:
                if kid in self._keys:
                    return self._keys[kid]  # serve stale rather than hammer the IdP
                raise OIDCError("unknown kid; JWKS refresh rate-limited")

            self._last_attempt = now
            try:
                keys = await self._fetch()
            except Exception as exc:  # IdP unreachable
                if kid in self._keys:
                    return self._keys[kid]  # serve through the outage
                raise OIDCError(f"JWKS fetch failed and kid not cached: {exc}") from exc

            self._keys = {k["kid"]: k for k in keys if "kid" in k}
            self._fetched_at = self._clock()
            if kid in self._keys:
                return self._keys[kid]
            raise OIDCError("unknown kid after JWKS refresh")


class OIDCValidator:
    """Validate an OIDC JWT and resolve it to a :class:`Principal`. Raises :class:`OIDCError`
    on any validation failure (the composite authenticator catches it)."""

    def __init__(self, cfg: OIDCConfig, roles: RoleScopes, jwks: JWKSResolver):
        self.cfg = cfg
        self.roles = roles
        self.jwks = jwks

    async def validate(self, token: str) -> Principal:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.InvalidTokenError as exc:
            raise OIDCError(f"malformed token header: {exc}") from exc

        alg = header.get("alg")
        if alg not in self.cfg.algorithms:  # rejects 'none', HS*, and unexpected algs
            raise OIDCError(f"algorithm not allowed: {alg!r}")
        kid = header.get("kid")
        if not kid:
            raise OIDCError("token header has no kid")

        jwk = await self.jwks.get_jwk(kid)  # only ever a key from the IdP's JWKS
        try:
            signing_key = jwt.PyJWK.from_dict(jwk).key
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=list(self.cfg.algorithms),
                audience=self.cfg.audience,
                issuer=self.cfg.issuer,
                leeway=self.cfg.leeway,
                options={"require": ["exp", "iss", "aud"]},
            )
        except jwt.InvalidTokenError as exc:
            raise OIDCError(f"token validation failed: {exc}") from exc
        except Exception as exc:  # malformed JWK, key errors, etc.
            raise OIDCError(f"token validation error: {exc}") from exc

        return self._principal_from_claims(claims)

    def _principal_from_claims(self, claims: Mapping) -> Principal:
        subject = claims.get(self.cfg.subject_claim) or claims.get("sub")
        if not subject:
            raise OIDCError("token has no subject claim")

        groups = claims.get(self.cfg.groups_claim, [])
        if isinstance(groups, str):
            groups = [groups]

        roles = [self.cfg.group_roles[g] for g in groups if g in self.cfg.group_roles]
        if roles:
            scopes = self.roles.scopes_for_roles(roles)
        elif self.cfg.default_role and self.cfg.default_role in self.roles:
            # Policy choice (consumer-set): an authenticated user with no mapped group gets a
            # minimal default role. With default_role=None this is instead zero scopes
            # (authenticated, every scoped route 403s — the audit still records who).
            scopes = self.roles.scopes_for_role(self.cfg.default_role)
        else:
            scopes = frozenset()

        return Principal(subject=f"oidc:{subject}", scopes=scopes, auth_method=AUTH_OIDC)


def _looks_like_jwt(token: str) -> bool:
    parts = token.split(".")
    return len(parts) == 3 and all(parts)


class CompositeAuthenticator:
    """OIDC JWT first; else a static/break-glass key; else ``None`` (-> 401 at the seam)."""

    def __init__(
        self,
        *,
        static: Optional[Authenticator] = None,
        oidc: Optional[OIDCValidator] = None,
    ):
        self._static = static
        self._oidc = oidc

    @property
    def enabled(self) -> bool:
        return bool(self._oidc) or bool(self._static and self._static.enabled)

    async def authenticate(self, token: Optional[str]) -> Optional[Principal]:
        if not token:
            return None
        principal: Optional[Principal] = None
        if self._oidc and _looks_like_jwt(token):
            try:
                principal = await self._oidc.validate(token)
            except OIDCError:
                principal = None  # fall through to break-glass keys
        if principal is None and self._static:
            principal = await self._static.authenticate(token)
        return principal


async def _discover_jwks_uri(client, issuer: str) -> str:
    doc = (await client.get(f"{issuer.rstrip('/')}/.well-known/openid-configuration")).json()
    uri = doc.get("jwks_uri")
    if not uri:
        raise AuthConfigError(f"issuer discovery doc has no jwks_uri: {issuer}")
    return uri


def _parse_group_roles(raw: str) -> dict[str, str]:
    """Parse ``"group-a:admin,group-b:user"`` into a dict."""
    out: dict[str, str] = {}
    for entry in (s.strip() for s in raw.split(",") if s.strip()):
        group, _, role = entry.partition(":")
        if group.strip() and role.strip():
            out[group.strip()] = role.strip()
    return out


def build_oidc_config(env: Optional[Mapping[str, str]] = None) -> Optional[OIDCConfig]:
    """Build :class:`OIDCConfig` from environment, or ``None`` if OIDC is disabled."""
    e = os.environ if env is None else env
    if e.get("OIDC_ENABLED", "").lower() not in ("1", "true", "yes"):
        return None
    issuer = e.get("OIDC_ISSUER")
    audience = e.get("OIDC_AUDIENCE")
    if not issuer or not audience:
        raise AuthConfigError("OIDC_ENABLED but OIDC_ISSUER / OIDC_AUDIENCE not set")
    algs = e.get("OIDC_ALGORITHMS", "RS256")
    return OIDCConfig(
        issuer=issuer,
        audience=audience,
        jwks_uri=e.get("OIDC_JWKS_URI") or None,
        groups_claim=e.get("OIDC_GROUPS_CLAIM", "groups"),
        subject_claim=e.get("OIDC_SUBJECT_CLAIM", "sub"),
        algorithms=frozenset(a.strip() for a in algs.split(",") if a.strip()),
        leeway=int(e.get("OIDC_LEEWAY", "60")),
        jwks_cache_ttl=int(e.get("OIDC_JWKS_CACHE_TTL", "600")),
        jwks_min_refresh_interval=int(e.get("OIDC_JWKS_MIN_REFRESH", "30")),
        group_roles=_parse_group_roles(e.get("OIDC_GROUP_ROLES", "")),
        default_role=e.get("OIDC_DEFAULT_ROLE") or None,
    )


def build_oidc_validator(
    roles: RoleScopes,
    *,
    env: Optional[Mapping[str, str]] = None,
    http_client=None,
) -> Optional[OIDCValidator]:
    """Build an :class:`OIDCValidator` from environment with a network-backed JWKS cache, or
    ``None`` if OIDC is disabled. ``http_client`` is an optional ``httpx.AsyncClient`` (created
    lazily if omitted). For offline tests, construct :class:`OIDCValidator` directly with a
    seeded :class:`JWKSCache` instead."""
    cfg = build_oidc_config(env)
    if cfg is None:
        return None

    async def fetch() -> list[dict]:
        import httpx  # local import: only needed for the network path

        client = http_client or httpx.AsyncClient(timeout=5.0)
        try:
            uri = cfg.jwks_uri or await _discover_jwks_uri(client, cfg.issuer)
            resp = await client.get(uri)
            resp.raise_for_status()
            return resp.json().get("keys", [])
        finally:
            if http_client is None:
                await client.aclose()

    jwks = JWKSCache(
        fetch,
        ttl=cfg.jwks_cache_ttl,
        min_refresh_interval=cfg.jwks_min_refresh_interval,
    )
    return OIDCValidator(cfg, roles, jwks)
