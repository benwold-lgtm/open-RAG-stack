import pytest

from rag_auth import ANONYMOUS, AUTH_NONE, Principal, RoleScopes, UnknownRole

# A representative consumer role map (chat-ui's), defined in the test — the module
# itself ships no domain scopes.
ROLES = RoleScopes(
    {
        "admin": {"chat:use", "convos:read", "convos:write", "models:read", "users:manage"},
        "user": {"chat:use", "convos:read", "convos:write", "models:read"},
    }
)


def test_scopes_for_role_returns_bundle():
    assert ROLES.scopes_for_role("user") == frozenset(
        {"chat:use", "convos:read", "convos:write", "models:read"}
    )
    assert "users:manage" in ROLES.scopes_for_role("admin")
    assert "users:manage" not in ROLES.scopes_for_role("user")


def test_unknown_role_raises():
    with pytest.raises(UnknownRole):
        ROLES.scopes_for_role("superuser")


def test_scopes_for_roles_is_union_and_ignores_unknown():
    # admin ∪ user == admin's bundle; an unknown IdP group is silently ignored.
    union = ROLES.scopes_for_roles(["user", "admin", "group-the-app-never-heard-of"])
    assert union == ROLES.scopes_for_role("admin")


def test_scopes_for_roles_empty():
    assert ROLES.scopes_for_roles([]) == frozenset()


def test_roles_and_all_scopes_properties():
    assert ROLES.roles == frozenset({"admin", "user"})
    assert ROLES.all_scopes == ROLES.scopes_for_role("admin")
    assert "user" in ROLES and "nope" not in ROLES


def test_empty_rolescopes():
    empty = RoleScopes({})
    assert empty.all_scopes == frozenset()
    assert empty.roles == frozenset()


def test_principal_has_and_has_any():
    p = Principal(subject="local:alice", scopes=frozenset({"chat:use"}), auth_method="local")
    assert p.has("chat:use")
    assert not p.has("users:manage")
    assert p.has_any(["users:manage", "chat:use"])
    assert not p.has_any(["users:manage", "settings:manage"])


def test_principal_is_frozen():
    p = Principal(subject="x", scopes=frozenset(), auth_method="local")
    with pytest.raises(Exception):
        p.subject = "y"  # type: ignore[misc]


def test_anonymous_has_no_scopes():
    assert ANONYMOUS.scopes == frozenset()
    assert ANONYMOUS.auth_method == AUTH_NONE
    assert not ANONYMOUS.has("chat:use")
