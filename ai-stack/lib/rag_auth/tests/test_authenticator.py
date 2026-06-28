import pytest

from rag_auth import RoleScopes, build_static_authenticator
from rag_auth.rbac import AUTH_LOCAL

ROLES = RoleScopes(
    {
        "admin": {"things:read", "things:write"},
        "viewer": {"things:read"},
    }
)


@pytest.mark.asyncio
async def test_break_glass_admin_key_resolves_to_admin():
    auth = build_static_authenticator(ROLES, env={"BREAK_GLASS_ADMIN_KEY": "secret-admin"})
    assert auth.enabled
    p = await auth.authenticate("secret-admin")
    assert p is not None
    assert p.auth_method == AUTH_LOCAL
    assert p.subject == "key:break-glass-admin"
    assert p.has("things:write")


@pytest.mark.asyncio
async def test_viewer_key_is_read_only():
    auth = build_static_authenticator(
        ROLES, env={"BREAK_GLASS_VIEWER_KEY": "ro"}, viewer_role="viewer"
    )
    p = await auth.authenticate("ro")
    assert p is not None
    assert p.has("things:read")
    assert not p.has("things:write")


@pytest.mark.asyncio
async def test_unknown_or_missing_token_returns_none():
    auth = build_static_authenticator(ROLES, env={"BREAK_GLASS_ADMIN_KEY": "secret-admin"})
    assert await auth.authenticate("wrong") is None
    assert await auth.authenticate(None) is None
    assert await auth.authenticate("") is None


@pytest.mark.asyncio
async def test_extra_keys_via_rag_auth_keys():
    auth = build_static_authenticator(
        ROLES, env={"RAG_AUTH_KEYS": "ci:viewer:cikey, ops:admin:opskey , bad-entry, x:nope:y"}
    )
    ci = await auth.authenticate("cikey")
    ops = await auth.authenticate("opskey")
    assert ci is not None and ci.subject == "key:ci" and not ci.has("things:write")
    assert ops is not None and ops.has("things:write")
    # malformed entry and unknown-role entry are skipped, not fatal
    assert await auth.authenticate("y") is None


def test_no_keys_means_disabled():
    auth = build_static_authenticator(ROLES, env={})
    assert not auth.enabled


def test_unknown_admin_role_is_skipped():
    # role name doesn't exist in the map -> key is not added (resilient to typos)
    roles = RoleScopes({"user": {"things:read"}})
    auth = build_static_authenticator(roles, env={"BREAK_GLASS_ADMIN_KEY": "k"})
    assert not auth.enabled
