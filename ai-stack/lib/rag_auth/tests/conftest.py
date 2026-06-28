"""Offline OIDC test helpers: a local RSA keypair, a JWKS built from its public key, and a
token signer. No network, no real IdP — everything is signed and verified in-process."""

import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

ISSUER = "https://idp.test/realms/corp"
AUDIENCE = "rag-stack"
KID = "test-kid-1"


@pytest.fixture(scope="session")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="session")
def jwks(rsa_key):
    """A JWKS 'keys' array containing the public half of rsa_key, tagged with KID."""
    pub_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(rsa_key.public_key()))
    pub_jwk.update({"kid": KID, "use": "sig", "alg": "RS256"})
    return [pub_jwk]


@pytest.fixture
def sign(rsa_key):
    """Return a signer: sign(claims, *, alg='RS256', kid=KID, key=rsa_key) -> token."""

    def _sign(claims, *, alg="RS256", kid=KID, key=None):
        headers = {"kid": kid} if kid is not None else {}
        return jwt.encode(claims, key or rsa_key, algorithm=alg, headers=headers)

    return _sign


@pytest.fixture
def claims_factory():
    """Return a factory producing a valid claim set with overridable fields."""

    def _make(**overrides):
        now = int(time.time())
        base = {
            "sub": "user-123",
            "iss": ISSUER,
            "aud": AUDIENCE,
            "iat": now,
            "exp": now + 300,
            "groups": [],
        }
        base.update(overrides)
        return base

    return _make
