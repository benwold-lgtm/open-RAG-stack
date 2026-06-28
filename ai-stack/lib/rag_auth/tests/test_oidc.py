"""Offline OIDC validation tests — the full threat matrix, no network.

Covers: valid, expired, wrong-aud, wrong-iss, bad-alg (none / HS256), unknown-kid,
no-mapped-group (default-deny and default-role), the rate-limited JWKS refetch, and the
IdP-down break-glass behaviour via the composite authenticator.
"""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from rag_auth import (
    AuthConfigError,
    CompositeAuthenticator,
    JWKSCache,
    OIDCConfig,
    OIDCError,
    OIDCValidator,
    RoleScopes,
    build_static_authenticator,
)

from conftest import AUDIENCE, ISSUER, KID

ROLES = RoleScopes(
    {
        "admin": {"chat:use", "convos:read", "convos:write", "users:manage"},
        "user": {"chat:use", "convos:read", "convos:write"},
    }
)


def make_cfg(**overrides) -> OIDCConfig:
    base = dict(
        issuer=ISSUER,
        audience=AUDIENCE,
        algorithms=frozenset({"RS256"}),
        group_roles={"app-admins": "admin", "app-users": "user"},
    )
    base.update(overrides)
    return OIDCConfig(**base)


def static_resolver(jwks):
    """A JWKSResolver that serves the seeded keys with no network and no expiry."""
    keys = {k["kid"]: k for k in jwks}

    class _R:
        async def get_jwk(self, kid):
            if kid not in keys:
                raise OIDCError("unknown kid")
            return keys[kid]

    return _R()


def validator(jwks, **cfg_overrides) -> OIDCValidator:
    return OIDCValidator(make_cfg(**cfg_overrides), ROLES, static_resolver(jwks))


# --- happy path + group→scope mapping ---------------------------------------

async def test_valid_admin_token_maps_groups_to_scopes(jwks, sign, claims_factory):
    token = sign(claims_factory(sub="alice", groups=["app-admins"]))
    p = await validator(jwks).validate(token)
    assert p.subject == "oidc:alice"
    assert p.auth_method == "oidc"
    assert p.has("users:manage")


async def test_multiple_groups_union_scopes(jwks, sign, claims_factory):
    token = sign(claims_factory(groups=["app-users", "app-admins"]))
    p = await validator(jwks).validate(token)
    assert p.has("users:manage") and p.has("chat:use")


async def test_string_groups_claim_is_accepted(jwks, sign, claims_factory):
    token = sign(claims_factory(groups="app-users"))
    p = await validator(jwks).validate(token)
    assert p.has("chat:use") and not p.has("users:manage")


# --- no-mapped-group policy --------------------------------------------------

async def test_unmapped_group_default_deny_zero_scopes(jwks, sign, claims_factory):
    token = sign(claims_factory(groups=["some-other-group"]))
    p = await validator(jwks).validate(token)  # default_role is None
    assert p.subject == "oidc:user-123"
    assert p.scopes == frozenset()  # authenticated, zero capability


async def test_unmapped_group_with_default_role(jwks, sign, claims_factory):
    token = sign(claims_factory(groups=[]))
    p = await validator(jwks, default_role="user").validate(token)
    assert p.has("chat:use") and not p.has("users:manage")


# --- rejection cases ---------------------------------------------------------

async def test_expired_token_rejected(jwks, sign, claims_factory):
    now = int(time.time())
    token = sign(claims_factory(iat=now - 600, exp=now - 300))
    with pytest.raises(OIDCError):
        await validator(jwks).validate(token)


async def test_wrong_audience_rejected(jwks, sign, claims_factory):
    token = sign(claims_factory(aud="some-other-service"))
    with pytest.raises(OIDCError):
        await validator(jwks).validate(token)


async def test_wrong_issuer_rejected(jwks, sign, claims_factory):
    token = sign(claims_factory(iss="https://evil.test/"))
    with pytest.raises(OIDCError):
        await validator(jwks).validate(token)


async def test_alg_none_rejected(jwks, claims_factory):
    # An unsigned 'none' token must never validate.
    token = jwt.encode(claims_factory(), key=None, algorithm="none", headers={"kid": KID})
    with pytest.raises(OIDCError):
        await validator(jwks).validate(token)


async def test_hs256_token_rejected(jwks, claims_factory):
    # Symmetric alg — the classic algorithm-confusion attack. Refused by the allow-list.
    token = jwt.encode(
        claims_factory(), key="x" * 32, algorithm="HS256", headers={"kid": KID}
    )
    with pytest.raises(OIDCError):
        await validator(jwks).validate(token)


async def test_token_signed_by_wrong_key_rejected(jwks, sign, claims_factory):
    attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = sign(claims_factory(), key=attacker_key)  # signed by attacker, KID claims to be ours
    with pytest.raises(OIDCError):
        await validator(jwks).validate(token)


async def test_missing_kid_rejected(jwks, sign, claims_factory):
    token = sign(claims_factory(), kid=None)
    with pytest.raises(OIDCError):
        await validator(jwks).validate(token)


async def test_unknown_kid_rejected(jwks, sign, claims_factory):
    token = sign(claims_factory(), kid="some-unknown-kid")
    with pytest.raises(OIDCError):
        await validator(jwks).validate(token)


async def test_missing_exp_rejected(jwks, sign, claims_factory):
    claims = claims_factory()
    del claims["exp"]
    token = sign(claims)
    with pytest.raises(OIDCError):
        await validator(jwks).validate(token)


# --- config-time safety ------------------------------------------------------

def test_config_rejects_symmetric_alg():
    with pytest.raises(AuthConfigError):
        make_cfg(algorithms=frozenset({"HS256"}))


def test_config_rejects_empty_algs():
    with pytest.raises(AuthConfigError):
        make_cfg(algorithms=frozenset())


# --- JWKS cache: bounded TTL + rate-limited refetch + outage -----------------

class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


async def test_jwks_cache_serves_from_cache_then_refreshes_on_ttl(jwks):
    clock = FakeClock()
    calls = {"n": 0}

    async def fetch():
        calls["n"] += 1
        return jwks

    cache = JWKSCache(fetch, ttl=600, min_refresh_interval=30, clock=clock)
    await cache.get_jwk(KID)  # first fetch
    await cache.get_jwk(KID)  # served from cache
    assert calls["n"] == 1
    clock.advance(601)  # TTL expired
    await cache.get_jwk(KID)  # refetch
    assert calls["n"] == 2


async def test_jwks_unknown_kid_refetch_is_rate_limited(jwks):
    clock = FakeClock()
    calls = {"n": 0}

    async def fetch():
        calls["n"] += 1
        return jwks

    cache = JWKSCache(fetch, ttl=600, min_refresh_interval=30, clock=clock)
    await cache.get_jwk(KID)  # populate (fetch #1, last attempt t=1000)
    assert calls["n"] == 1
    clock.advance(31)  # move past the refresh window
    # An unknown kid triggers exactly one refetch; a second unknown kid inside the window
    # is rate-limited and does NOT fetch again (unknown-kid flood ≠ fetch flood).
    with pytest.raises(OIDCError):
        await cache.get_jwk("unknown-1")  # refetch #2 (still unknown afterwards)
    with pytest.raises(OIDCError):
        await cache.get_jwk("unknown-2")  # rate-limited, NO fetch
    assert calls["n"] == 2


async def test_jwks_serves_stale_through_idp_outage(jwks):
    clock = FakeClock()
    state = {"fail": False, "n": 0}

    async def fetch():
        state["n"] += 1
        if state["fail"]:
            raise RuntimeError("IdP down")
        return jwks

    cache = JWKSCache(fetch, ttl=600, min_refresh_interval=0, clock=clock)
    await cache.get_jwk(KID)  # warm the cache
    state["fail"] = True
    clock.advance(601)  # TTL expired → will try to refetch and fail
    # Known kid is still served from cache through the outage.
    assert (await cache.get_jwk(KID))["kid"] == KID


# --- composite: OIDC → static → 401, with IdP-down break-glass ---------------

async def test_composite_prefers_oidc_then_falls_back(jwks, sign, claims_factory):
    static = build_static_authenticator(ROLES, env={"BREAK_GLASS_ADMIN_KEY": "bg-admin"})
    comp = CompositeAuthenticator(static=static, oidc=validator(jwks))

    # A valid JWT → OIDC path.
    jwt_token = sign(claims_factory(groups=["app-users"]))
    p = await comp.authenticate(jwt_token)
    assert p is not None and p.auth_method == "oidc"

    # A non-JWT bearer → static path.
    p2 = await comp.authenticate("bg-admin")
    assert p2 is not None and p2.auth_method == "local" and p2.has("users:manage")

    # Garbage → 401 (None).
    assert await comp.authenticate("nonsense") is None
    assert await comp.authenticate(None) is None


async def test_composite_idp_down_break_glass_still_works(jwks, sign, claims_factory):
    # OIDC validator whose JWKS resolver always fails (IdP unreachable, kid not cached).
    class DeadResolver:
        async def get_jwk(self, kid):
            raise OIDCError("IdP unreachable")

    dead_oidc = OIDCValidator(make_cfg(), ROLES, DeadResolver())
    static = build_static_authenticator(ROLES, env={"BREAK_GLASS_ADMIN_KEY": "bg-admin"})
    comp = CompositeAuthenticator(static=static, oidc=dead_oidc)

    # A real JWT can't be validated (IdP down) → OIDC fails closed...
    jwt_token = sign(claims_factory(groups=["app-admins"]))
    assert await comp.authenticate(jwt_token) is None
    # ...but the break-glass admin key still gets you in.
    p = await comp.authenticate("bg-admin")
    assert p is not None and p.has("users:manage")


# --- validate_id_token (RP login flow: returns claims, enforces nonce) -------

async def test_validate_id_token_returns_claims(jwks, sign, claims_factory):
    token = sign(claims_factory(sub="alice", nonce="n-123", email="a@corp.test"))
    claims = await validator(jwks).validate_id_token(token, nonce="n-123")
    assert claims["sub"] == "alice" and claims["email"] == "a@corp.test"


async def test_validate_id_token_nonce_mismatch_rejected(jwks, sign, claims_factory):
    token = sign(claims_factory(nonce="issued"))
    with pytest.raises(OIDCError):
        await validator(jwks).validate_id_token(token, nonce="different")
