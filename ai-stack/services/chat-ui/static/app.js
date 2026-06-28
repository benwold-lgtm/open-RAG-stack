/* chat-ui SPA — vanilla JS, no framework. Auth gate → login (local + SSO) → chat → admin.
   Talks only to chat-ui's own /api; markdown is rendered with the vendored marked + DOMPurify. */
"use strict";

const app = document.getElementById("app");
const BRAND = app.dataset.brand || "Open RAG Chat";

const state = {
  config: null,      // /api/auth/config
  me: null,          // /api/auth/me  {username, role, scopes:[]}
  convos: [],        // sidebar list
  activeId: null,    // open conversation id
  sending: false,
};

const can = (scope) => state.me && state.me.scopes.includes(scope);
const esc = (s) => s.replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const renderMarkdown = (md) =>
  DOMPurify.sanitize(marked.parse(md || "", { breaks: true }));

// ── tiny API layer (cookies ride along automatically) ────────────────────────
async function api(path, { method = "GET", body } = {}) {
  const opt = { method, headers: {} };
  if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  let data = null;
  try { data = await r.json(); } catch (_) { /* empty body */ }
  if (!r.ok) throw { status: r.status, detail: (data && data.detail) || r.statusText };
  return data;
}

// ── boot ─────────────────────────────────────────────────────────────────────
async function boot() {
  try { state.config = await api("/api/auth/config"); } catch (_) { state.config = { local_login: true, oidc: { enabled: false } }; }
  document.title = (state.config.brand_name) || BRAND;
  try {
    state.me = await api("/api/auth/me");
    await openChat();
  } catch (e) {
    renderLogin();
  }
}

// ── login / register ─────────────────────────────────────────────────────────
function renderLogin(notice) {
  const cfg = state.config;
  const local = cfg.local_login !== false;
  const reg = !!cfg.registration_enabled;
  app.className = "";
  app.innerHTML = `
    <div class="login-wrap"><div class="login-card">
      <div class="dot"></div>
      <h1>${esc(cfg.brand_name || BRAND)}</h1>
      <p class="sub">Sign in to start chatting with your documents.</p>
      <div id="notice"></div>
      ${cfg.oidc && cfg.oidc.enabled ? `<button class="sso" id="sso">Sign in with SSO</button>` : ""}
      ${cfg.oidc && cfg.oidc.enabled && local ? `<div class="divider">or</div>` : ""}
      ${local ? `
        <form id="loginForm">
          <div class="field"><label>Username</label><input name="username" autocomplete="username" required></div>
          <div class="field"><label>Password</label><input name="password" type="password" autocomplete="current-password" required></div>
          <button class="primary" type="submit">Sign in</button>
        </form>
        ${reg ? `<p class="toggle">New here? <button id="toReg">Create an account</button></p>` : ""}
      ` : (!cfg.oidc || !cfg.oidc.enabled ? `<p class="sub">No sign-in methods are enabled. Contact your administrator.</p>` : "")}
    </div></div>`;
  if (notice) showNotice(notice.kind, notice.text);

  const sso = document.getElementById("sso");
  if (sso) sso.onclick = () => { window.location.href = cfg.oidc.login_path || "/api/auth/oidc/login"; };

  const form = document.getElementById("loginForm");
  if (form) form.onsubmit = async (e) => {
    e.preventDefault();
    const f = new FormData(form);
    try {
      state.me = await api("/api/auth/login", { method: "POST", body: { username: f.get("username"), password: f.get("password") } });
      state.me = await api("/api/auth/me");
      await openChat();
    } catch (err) { showNotice("err", loginError(err)); }
  };
  const toReg = document.getElementById("toReg");
  if (toReg) toReg.onclick = renderRegister;
}

function loginError(err) {
  if (err.status === 429) return "Too many attempts. Please wait a moment and try again.";
  if (err.status === 403) return err.detail || "This account isn't allowed to sign in yet.";
  return "Invalid username or password.";
}

function renderRegister() {
  app.innerHTML = `
    <div class="login-wrap"><div class="login-card">
      <div class="dot"></div>
      <h1>Create your account</h1>
      <p class="sub">Pick a username and a password (at least 8 characters).</p>
      <div id="notice"></div>
      <form id="regForm">
        <div class="field"><label>Username</label><input name="username" autocomplete="username" required></div>
        <div class="field"><label>Email (optional)</label><input name="email" type="email" autocomplete="email"></div>
        <div class="field"><label>Password</label><input name="password" type="password" autocomplete="new-password" required></div>
        <button class="primary" type="submit">Create account</button>
      </form>
      <p class="toggle">Already have an account? <button id="toLogin">Sign in</button></p>
    </div></div>`;
  document.getElementById("toLogin").onclick = () => renderLogin();
  document.getElementById("regForm").onsubmit = async (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    const body = { username: f.get("username"), password: f.get("password") };
    if (f.get("email")) body.email = f.get("email");
    try {
      const res = await api("/api/auth/register", { method: "POST", body });
      if (res && res.status === "active") { state.me = await api("/api/auth/me"); await openChat(); }
      else renderLogin({ kind: "ok", text: "Account created. An administrator needs to approve it before you can sign in." });
    } catch (err) {
      showNotice("err", err.status === 409 ? "That username is taken." :
        err.status === 422 ? (err.detail || "Please check your details.") :
        err.status === 403 ? "Registration is disabled." : "Could not create the account.");
    }
  };
}

function showNotice(kind, text) {
  const n = document.getElementById("notice");
  if (n) n.innerHTML = `<div class="msg ${kind === "ok" ? "ok" : "err"}">${esc(text)}</div>`;
}

// ── app shell ─────────────────────────────────────────────────────────────────
function renderShell(mainHtml) {
  app.className = "app";
  app.innerHTML = `
    <div class="topbar">
      <div class="brand"><span class="dot"></span>${esc(state.config.brand_name || BRAND)}</div>
      <div class="spacer"></div>
      ${can("users:manage") ? `<button class="ghost" id="navAdmin">Admin</button><button class="ghost" id="navChat">Chat</button>` : ""}
      <span class="who">${esc(state.me.username)}</span>
      <button class="ghost" id="logout">Log out</button>
    </div>
    <div class="sidebar">
      <div class="new"><button class="primary" id="newChat">+ New chat</button></div>
      <div class="convos" id="convos"></div>
    </div>
    <div class="main" id="mainPane">${mainHtml}</div>`;
  document.getElementById("logout").onclick = async () => { try { await api("/api/auth/logout", { method: "POST" }); } catch (_) {} state.me = null; renderLogin(); };
  document.getElementById("newChat").onclick = newConversation;
  const a = document.getElementById("navAdmin"); if (a) a.onclick = openAdmin;
  const cc = document.getElementById("navChat"); if (cc) cc.onclick = () => openChat();
  renderConvos();
}

// ── conversations sidebar ──────────────────────────────────────────────────────
async function refreshConvos() {
  try { state.convos = await api("/api/conversations"); } catch (_) { state.convos = []; }
}

function renderConvos() {
  const box = document.getElementById("convos");
  if (!box) return;
  if (!state.convos.length) { box.innerHTML = `<div class="empty">No conversations yet.<br>Start a new chat.</div>`; return; }
  box.innerHTML = "";
  for (const c of state.convos) {
    const row = document.createElement("div");
    row.className = "convo" + (c.id === state.activeId ? " active" : "");
    row.innerHTML = `<button class="title">${esc(c.title || "Untitled chat")}</button>
      <button class="row-act" title="Rename">✎</button><button class="row-act" title="Delete">🗑</button>`;
    const [titleBtn, renameBtn, delBtn] = row.querySelectorAll("button");
    titleBtn.onclick = () => openConversation(c.id);
    renameBtn.onclick = (e) => { e.stopPropagation(); renameConversation(c); };
    delBtn.onclick = (e) => { e.stopPropagation(); deleteConversation(c); };
    box.appendChild(row);
  }
}

async function newConversation() {
  try {
    const c = await api("/api/conversations", { method: "POST", body: {} });
    await refreshConvos();
    openConversation(c.id);
  } catch (_) {}
}

async function renameConversation(c) {
  const title = prompt("Rename conversation", c.title || "");
  if (title === null) return;
  try { await api(`/api/conversations/${c.id}`, { method: "PATCH", body: { title } }); await refreshConvos(); renderConvos(); } catch (_) {}
}

async function deleteConversation(c) {
  if (!confirm(`Delete "${c.title || "this conversation"}"? This cannot be undone.`)) return;
  try {
    await api(`/api/conversations/${c.id}`, { method: "DELETE" });
    if (state.activeId === c.id) state.activeId = null;
    await refreshConvos();
    state.activeId ? renderConvos() : openChat();
  } catch (_) {}
}

// ── chat view ──────────────────────────────────────────────────────────────────
async function openChat() {
  await refreshConvos();
  renderShell(`<div class="messages" id="messages"></div>`);
  state.activeId = null;
  document.getElementById("messages").innerHTML = `
    <div class="welcome"><h2>Welcome${state.me ? ", " + esc(state.me.username) : ""}</h2>
    <p>Ask a question about your documents to get started. Answers come with page-level citations.</p></div>`;
  renderComposer();
}

async function openConversation(id) {
  state.activeId = id;
  renderShell(`<div class="messages" id="messages"></div>`);
  renderConvos();
  renderComposer();
  const box = document.getElementById("messages");
  box.innerHTML = `<div class="empty">Loading…</div>`;
  try {
    const conv = await api(`/api/conversations/${id}`);
    box.innerHTML = "";
    if (!conv.messages.length) box.innerHTML = `<div class="welcome"><p>No messages yet — say hello below.</p></div>`;
    for (const m of conv.messages) addBubble(m.role, m.content);
  } catch (_) { box.innerHTML = `<div class="empty">Could not load this conversation.</div>`; }
}

function renderComposer() {
  const pane = document.getElementById("mainPane");
  if (pane.querySelector(".composer")) return;
  const bar = document.createElement("div");
  bar.className = "composer";
  bar.innerHTML = `<textarea id="input" rows="1" placeholder="Ask a question…"></textarea>
    <button class="primary" id="send">Send</button>`;
  pane.appendChild(bar);
  const ta = bar.querySelector("#input");
  ta.addEventListener("input", () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 160) + "px"; });
  ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
  bar.querySelector("#send").onclick = send;
}

function addBubble(role, content) {
  const box = document.getElementById("messages");
  const welcome = box.querySelector(".welcome, .empty"); if (welcome) welcome.remove();
  const b = document.createElement("div");
  b.className = `bubble ${role}`;
  if (role === "assistant") b.innerHTML = renderMarkdown(content);
  else b.textContent = content;
  box.appendChild(b);
  box.scrollTop = box.scrollHeight;
  return b;
}

async function send() {
  if (state.sending) return;
  const ta = document.getElementById("input");
  const text = ta.value.trim();
  if (!text) return;

  // ensure there's an active conversation to post into
  if (!state.activeId) {
    try { const c = await api("/api/conversations", { method: "POST", body: {} }); state.activeId = c.id; await refreshConvos(); renderConvos(); }
    catch (_) { return; }
  }

  state.sending = true;
  ta.value = ""; ta.style.height = "auto";
  document.getElementById("send").disabled = true;
  addBubble("user", text);
  const bubble = addBubble("assistant", "");
  bubble.classList.add("pending");
  bubble.textContent = "Thinking…";

  let answer = "";
  try {
    const resp = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: state.activeId, content: text }),
    });
    if (!resp.ok) {
      let detail = "The assistant is unavailable right now.";
      try { detail = (await resp.json()).detail || detail; } catch (_) {}
      bubble.classList.remove("pending"); bubble.textContent = "⚠ " + detail;
      throw new Error("chat failed");
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 2);
        if (!line.startsWith("data:")) continue;
        const evt = JSON.parse(line.slice(5).trim());
        if (evt.delta != null) {
          answer += evt.delta;
          bubble.classList.remove("pending");
          bubble.innerHTML = renderMarkdown(answer);
          document.getElementById("messages").scrollTop = 1e9;
        }
        if (evt.done) { await refreshConvos(); renderConvos(); }
      }
    }
  } catch (_) {
    if (!answer) { bubble.classList.remove("pending"); if (!bubble.textContent.startsWith("⚠")) bubble.textContent = "⚠ Something went wrong. Please try again."; }
  } finally {
    state.sending = false;
    const sb = document.getElementById("send"); if (sb) sb.disabled = false;
  }
}

// ── admin ──────────────────────────────────────────────────────────────────────
async function openAdmin() {
  if (!can("users:manage")) return;
  renderShell(`<div class="admin" id="admin"><h2>User management</h2><div id="users">Loading…</div></div>`);
  await loadUsers();
}

async function loadUsers() {
  const box = document.getElementById("users");
  let users;
  try { users = await api("/api/admin/users"); } catch (_) { box.innerHTML = `<div class="empty">Could not load users.</div>`; return; }
  const rows = users.map((u) => {
    const acts = [];
    if (u.status === "pending") acts.push(`<button data-act="approve" data-id="${u.id}">Approve</button>`);
    if (u.status !== "disabled") acts.push(`<button data-act="disable" data-id="${u.id}">Disable</button>`);
    acts.push(u.role === "admin"
      ? `<button data-act="role" data-id="${u.id}" data-role="user">Make user</button>`
      : `<button data-act="role" data-id="${u.id}" data-role="admin">Make admin</button>`);
    return `<tr>
      <td>${esc(u.username)}</td>
      <td><span class="pill">${esc(u.auth_source)}</span></td>
      <td>${esc(u.role)}</td>
      <td><span class="pill ${u.status}">${esc(u.status)}</span></td>
      <td><div class="acts">${acts.join("")}</div></td></tr>`;
  }).join("");
  box.innerHTML = `<table>
    <thead><tr><th>Username</th><th>Source</th><th>Role</th><th>Status</th><th>Actions</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
  box.querySelectorAll("button[data-act]").forEach((btn) => {
    btn.onclick = () => adminAction(btn.dataset.act, btn.dataset.id, btn.dataset.role);
  });
}

async function adminAction(act, id, role) {
  try {
    if (act === "approve") await api(`/api/admin/users/${id}/approve`, { method: "POST" });
    else if (act === "disable") await api(`/api/admin/users/${id}/disable`, { method: "POST" });
    else if (act === "role") await api(`/api/admin/users/${id}/role`, { method: "POST", body: { role } });
    await loadUsers();
  } catch (err) {
    alert(err.status === 409 ? (err.detail || "That action would remove the last admin.") : "Action failed.");
  }
}

boot();
