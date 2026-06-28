"""rag_auth — shared auth framework for open-RAG-stack services.

Public surface (built up across A1-A4):

* Scope/role model:    Principal, RoleScopes, ANONYMOUS, AUTH_* constants
* Authenticators:      Authenticator, StaticKeyAuthenticator, build_static_authenticator,
                       CompositeAuthenticator
* OIDC:                OIDCConfig, OIDCValidator, JWKSCache, build_oidc_validator
* FastAPI seam:        make_authenticate_request, require_scope, current_principal,
                       verify_boot_config
* Whoami contract:     whoami_payload, make_whoami_router
* Errors:              AuthConfigError, OIDCError, UnknownRole
"""

from .authenticator import (
    Authenticator,
    StaticKeyAuthenticator,
    build_static_authenticator,
)
from .deps import (
    current_principal,
    make_authenticate_request,
    require_scope,
    verify_boot_config,
)
from .errors import AuthConfigError, OIDCError, UnknownRole
from .oidc import (
    ASYMMETRIC_ALGS,
    CompositeAuthenticator,
    JWKSCache,
    OIDCConfig,
    OIDCValidator,
    build_oidc_config,
    build_oidc_validator,
)
from .rbac import (
    AUTH_LOCAL,
    AUTH_NONE,
    AUTH_OIDC,
    ANONYMOUS,
    Principal,
    RoleScopes,
)
from .whoami import make_whoami_router, whoami_payload

__all__ = [
    # rbac
    "AUTH_LOCAL",
    "AUTH_OIDC",
    "AUTH_NONE",
    "ANONYMOUS",
    "Principal",
    "RoleScopes",
    # authenticator
    "Authenticator",
    "StaticKeyAuthenticator",
    "build_static_authenticator",
    # oidc
    "ASYMMETRIC_ALGS",
    "CompositeAuthenticator",
    "JWKSCache",
    "OIDCConfig",
    "OIDCValidator",
    "build_oidc_config",
    "build_oidc_validator",
    # deps
    "make_authenticate_request",
    "require_scope",
    "current_principal",
    "verify_boot_config",
    # whoami
    "whoami_payload",
    "make_whoami_router",
    # errors
    "AuthConfigError",
    "OIDCError",
    "UnknownRole",
]
