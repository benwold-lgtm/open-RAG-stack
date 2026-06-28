import os
import base64
import secrets
import httpx
from urllib.parse import quote
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response

INGESTION_URL = os.environ.get("INGESTION_URL", "http://ingestion:8002")

# Optional HTTP Basic Auth (Phase 7.P6). Off unless BOTH are set, so existing
# deploys are unaffected. When on, every route except /health requires the creds.
ADMIN_USER = os.environ.get("ADMIN_USER", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
AUTH_ENABLED = bool(ADMIN_USER and ADMIN_PASSWORD)

app = FastAPI(docs_url=None, redoc_url=None)


@app.middleware("http")
async def _basic_auth(request: Request, call_next):
    # /health stays open so container/k8s probes never need credentials.
    if AUTH_ENABLED and request.url.path != "/health":
        ok = False
        header = request.headers.get("authorization", "")
        if header.startswith("Basic "):
            try:
                user, _, pw = base64.b64decode(header[6:]).decode("utf-8").partition(":")
                ok = (secrets.compare_digest(user, ADMIN_USER)
                      and secrets.compare_digest(pw, ADMIN_PASSWORD))
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="RAG Admin"'})
    return await call_next(request)

# ---------------------------------------------------------------------------
# HTML page — served at GET /
# Future work: contributor roles (per-user write access), ingestion audit log.
# Current access model: LAN-only, no authentication. Anyone who can reach
# port 8005 can add or delete content. Control access at the network level.
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Admin — Open RAG Stack</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#f1f5f9;color:#1e293b;min-height:100vh}
nav{background:#1e293b;color:#fff;padding:.75rem 1.5rem;display:flex;align-items:center;gap:1rem}
nav h1{font-size:1rem;font-weight:600;letter-spacing:.02em}
.badge{background:#334155;color:#94a3b8;font-size:.7rem;padding:.2rem .5rem;border-radius:4px}
main{display:grid;grid-template-columns:390px 1fr;gap:1rem;padding:1rem;max-width:1600px}
.card{background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.1);padding:1.25rem}
.card h2{font-size:.8rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#64748b;margin-bottom:1rem}
.field{margin-bottom:.75rem}
.field label{display:block;font-size:.8rem;font-weight:500;color:#475569;margin-bottom:.3rem}
input,select{width:100%;padding:.45rem .7rem;border:1px solid #e2e8f0;border-radius:6px;font-size:.875rem;background:#fff;color:#1e293b}
input:focus,select:focus{outline:none;border-color:#2563eb;box-shadow:0 0 0 2px rgba(37,99,235,.1)}
.btn{padding:.45rem .9rem;border:none;border-radius:6px;font-size:.85rem;font-weight:500;cursor:pointer;transition:opacity .15s}
.btn:hover{opacity:.82}
.btn-primary{background:#2563eb;color:#fff}
.btn-ghost{background:#e2e8f0;color:#475569}
.btn-danger{background:#fee2e2;color:#dc2626}
.btn-sm{padding:.25rem .55rem;font-size:.75rem}
.inline{display:flex;gap:.5rem;align-items:flex-end}
.inline .field{flex:1;margin-bottom:0}
.drop-zone{border:2px dashed #cbd5e1;border-radius:8px;padding:2rem 1rem;text-align:center;cursor:pointer;transition:border-color .15s,background .15s;margin-bottom:.75rem}
.drop-zone:hover,.drop-zone.over{border-color:#2563eb;background:#eff6ff}
.drop-icon{font-size:2rem;margin-bottom:.4rem;color:#94a3b8}
.drop-zone p{font-size:.78rem;color:#64748b;line-height:1.6}
.drop-zone p strong{color:#2563eb}
#file-input{display:none}
.divider{display:flex;align-items:center;gap:.75rem;color:#94a3b8;font-size:.75rem;margin:.9rem 0}
.divider::before,.divider::after{content:'';flex:1;border-top:1px solid #e2e8f0}
.queue{margin-top:.6rem}
.qi{display:flex;align-items:center;gap:.5rem;padding:.35rem 0;border-bottom:1px solid #f1f5f9;font-size:.78rem}
.qi .qname{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#334155}
.sb{padding:.15rem .45rem;border-radius:9999px;font-size:.68rem;font-weight:500;white-space:nowrap}
.s-pending{background:#fef9c3;color:#854d0e}
.s-processing{background:#dbeafe;color:#1d4ed8}
.s-done,.s-completed{background:#dcfce7;color:#166534}
.s-error,.s-failed{background:#fee2e2;color:#991b1b}
.s-uploading{background:#e0e7ff;color:#3730a3}
.s-unchanged{background:#f1f5f9;color:#64748b}
.docs-hd{display:flex;align-items:center;gap:.5rem;margin-bottom:1rem}
.docs-hd h2{margin:0}
.sp{flex:1}
.docs-hd select{width:auto;padding:.3rem .55rem}
table{width:100%;border-collapse:collapse;font-size:.78rem}
th{text-align:left;padding:.4rem .6rem;color:#64748b;font-weight:500;border-bottom:2px solid #e2e8f0;white-space:nowrap}
td{padding:.45rem .6rem;border-bottom:1px solid #f1f5f9;vertical-align:middle}
tr:hover td{background:#f8fafc}
.src{max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.toasts{position:fixed;bottom:1rem;right:1rem;display:flex;flex-direction:column;gap:.5rem;z-index:100}
.toast{background:#1e293b;color:#fff;padding:.55rem .9rem;border-radius:6px;font-size:.78rem;box-shadow:0 4px 12px rgba(0,0,0,.2);animation:si .2s ease}
.toast.success{border-left:3px solid #22c55e}
.toast.error{border-left:3px solid #ef4444}
.toast.info{border-left:3px solid #3b82f6}
@keyframes si{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}
.note{font-size:.72rem;color:#94a3b8;margin-top:.75rem;line-height:1.6;padding:.6rem .75rem;background:#f8fafc;border-radius:6px;border-left:3px solid #e2e8f0}
details summary{font-size:.8rem;color:#64748b;cursor:pointer;user-select:none;padding:.25rem 0}
details .inner{margin-top:.75rem}
/* sortable headers, filter, error modal (Phase 7 P2/P3) */
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:#2563eb}
th .arrow{font-size:.7em;margin-left:.15rem;color:#2563eb}
.docs-hd input{width:160px}
.sb.clickable{cursor:pointer;text-decoration:underline dotted}
.sb.clickable:hover{filter:brightness(.95)}
.modal-bg{position:fixed;inset:0;background:rgba(15,23,42,.45);display:none;align-items:center;justify-content:center;z-index:200}
.modal-bg.show{display:flex}
.modal{background:#fff;border-radius:8px;max-width:580px;width:90%;max-height:80vh;overflow:auto;padding:1.4rem;box-shadow:0 12px 40px rgba(0,0,0,.3)}
.modal h3{font-size:1rem;color:#dc2626;margin-bottom:.75rem}
.modal .meta{font-size:.8rem;color:#475569;line-height:1.7;margin-bottom:.5rem}
.modal .meta b{color:#1e293b}
.modal pre{background:#0f172a;color:#fca5a5;padding:.8rem;border-radius:6px;font-size:.74rem;white-space:pre-wrap;word-break:break-word;margin:.5rem 0 1rem;font-family:ui-monospace,monospace}
/* inline field help (Phase 7.7 — deep-crawl guidance) */
.help{font-size:.72rem;color:#94a3b8;line-height:1.5;margin-top:.3rem}
.help b{color:#475569;font-weight:600}
.help code{background:#f1f5f9;color:#475569;padding:.05rem .3rem;border-radius:4px;font-size:.95em}
.help a{color:#2563eb;text-decoration:none}
.help a:hover{text-decoration:underline}
.help-intro{font-size:.76rem;color:#64748b;line-height:1.6;margin-bottom:.8rem}
/* bulk selection + actions (Phase 7.8) */
.bulk-bar{display:none;align-items:center;gap:.5rem;background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:.45rem .7rem;margin-bottom:.6rem;font-size:.8rem;color:#334155}
.bulk-bar b{color:#1d4ed8}
th.cbcol,td.cbcol{width:30px;text-align:center;padding-left:.4rem;padding-right:.2rem}
.cbcol input,#select-all{cursor:pointer;width:15px;height:15px;vertical-align:middle}
</style>
</head>
<body>
<nav>
  <h1>Open RAG Stack &mdash; Admin</h1>
  <span class="badge"><!--AUTH_BADGE--></span>
</nav>
<main>
  <!-- ── Left: ingest panel ─────────────────────────────────────────── -->
  <div>
    <div class="card">
      <h2>Add Content</h2>

      <div class="field">
        <label>Collection</label>
        <div class="inline">
          <div class="field"><select id="col-sel"><option value="">Loading&hellip;</option></select></div>
          <button class="btn btn-ghost btn-sm" onclick="toggleNewCol()">+ New</button>
          <button class="btn btn-ghost btn-sm" onclick="toggleRenameCol()">Rename</button>
        </div>
      </div>
      <div class="field" id="new-col-row" style="display:none">
        <label>New collection name</label>
        <div style="display:flex;gap:.5rem">
          <input type="text" id="new-col-input" placeholder="e.g. company-policies">
          <button class="btn btn-primary btn-sm" onclick="createCollection()">Create</button>
        </div>
      </div>
      <div class="field" id="rename-col-row" style="display:none">
        <label>Rename selected collection</label>
        <div style="display:flex;gap:.5rem">
          <input type="text" id="rename-col-input" placeholder="new name">
          <button class="btn btn-primary btn-sm" onclick="renameCollection()">Rename</button>
        </div>
        <p class="help">Moves all of this collection's documents and vectors to the new name. May take a moment for large collections.</p>
      </div>

      <div class="field">
        <label>Vendor / source tag</label>
        <input type="text" id="vendor" placeholder="e.g. hr, legal, engineering">
      </div>

      <div class="drop-zone" id="dz" onclick="document.getElementById('file-input').click()">
        <div class="drop-icon">&#128196;</div>
        <p><strong>Drop files here</strong> or click to browse</p>
        <p>PDF &nbsp;&middot;&nbsp; DOCX &nbsp;&middot;&nbsp; PPTX &nbsp;&middot;&nbsp; TXT &nbsp;&middot;&nbsp; MD</p>
      </div>
      <input type="file" id="file-input" multiple accept=".pdf,.docx,.pptx,.txt,.md">
      <div class="queue" id="queue"></div>

      <div class="divider">or ingest from URL</div>

      <div class="field">
        <label>URL</label>
        <div style="display:flex;gap:.5rem">
          <input type="text" id="url-in" placeholder="https://&hellip;">
          <button class="btn btn-primary btn-sm" onclick="ingestUrl()">Ingest</button>
        </div>
      </div>

      <details>
        <summary>&#128376; Deep crawl options</summary>
        <div class="inner">
          <p class="help-intro">Turn one starting URL into many documents: the crawler follows the links on that page (and the pages they lead to) and ingests each one &mdash; ideal for pulling in a whole documentation site or product section at once. Start with the defaults and widen if you need more.</p>
          <div style="display:flex;gap:.5rem">
            <div class="field" style="flex:1">
              <label>Max depth</label>
              <input type="number" id="depth" value="2" min="1" max="5">
              <p class="help">How many link-hops to follow. <b>1</b> = only pages linked directly from your URL; <b>2</b> = those plus the pages they link to. Higher = broader but slower (1&ndash;5).</p>
            </div>
            <div class="field" style="flex:1">
              <label>Max pages</label>
              <input type="number" id="pages" value="30" min="1" max="200">
              <p class="help">A hard cap on the total pages fetched, so a crawl can&rsquo;t run away on a large site (1&ndash;200).</p>
            </div>
          </div>
          <div class="field">
            <label>URL pattern filter (optional)</label>
            <input type="text" id="pattern" placeholder="e.g. */docs/*">
            <p class="help">Only crawl links matching this wildcard (<code>*</code> = any characters). Example: <code>*/docs/*</code> follows <code>&hellip;/docs/install</code> but skips <code>&hellip;/blog/&hellip;</code> and <code>&hellip;/pricing</code>. Leave blank to follow every link, within the limits above.</p>
          </div>
          <button class="btn btn-ghost" style="width:100%" onclick="deepCrawl()">Start deep crawl from URL above</button>
          <p class="help" style="text-align:center;margin-top:.5rem">&#128214; Full guide &amp; examples: <a href="https://github.com/benwold-lgtm/open-RAG-stack#deep-crawl-explained" target="_blank" rel="noopener">README &rarr; Deep crawl explained</a></p>
        </div>
      </details>

      <div class="note">
        <!--AUTH_NOTE-->
      </div>
    </div>
  </div>

  <!-- ── Right: document list ──────────────────────────────────────── -->
  <div class="card">
    <div class="docs-hd">
      <h2>Documents</h2>
      <div class="sp"></div>
      <input type="text" id="filter-text" placeholder="Filter&hellip;" oninput="renderDocs()">
      <select id="filter-col" onchange="loadDocs()">
        <option value="">All collections</option>
      </select>
      <button class="btn btn-ghost btn-sm" onclick="loadDocs()">&#8635; Refresh</button>
    </div>
    <div class="bulk-bar" id="bulk-bar">
      <span id="bulk-count"><b>0</b> selected</span>
      <div class="sp"></div>
      <button class="btn btn-ghost btn-sm" onclick="showBulkMove()">&#8631; Move to&hellip;</button>
      <button class="btn btn-ghost btn-sm" onclick="showBulkVendor()">&#9998; Set vendor&hellip;</button>
      <button class="btn btn-danger btn-sm" onclick="bulkDelete()">&#x2715; Delete</button>
      <button class="btn btn-ghost btn-sm" onclick="clearSelection()">Clear</button>
    </div>
    <table>
      <thead>
        <tr>
          <th class="cbcol"><input type="checkbox" id="select-all" title="Select all (filtered)"></th>
          <th class="sortable" onclick="sortBy('src')">Source<span class="arrow" data-k="src"></span></th>
          <th class="sortable" onclick="sortBy('collection')">Collection<span class="arrow" data-k="collection"></span></th>
          <th class="sortable" onclick="sortBy('vendor')">Vendor<span class="arrow" data-k="vendor"></span></th>
          <th class="sortable" onclick="sortBy('source_type')">Type<span class="arrow" data-k="source_type"></span></th>
          <th class="sortable" onclick="sortBy('status')">Status<span class="arrow" data-k="status"></span></th>
          <th class="sortable" onclick="sortBy('updated_at')">Updated<span class="arrow" data-k="updated_at"></span></th>
          <th></th>
        </tr>
      </thead>
      <tbody id="docs-body">
        <tr><td colspan="8" style="text-align:center;color:#94a3b8;padding:2rem">Loading&hellip;</td></tr>
      </tbody>
    </table>
  </div>
</main>
<div class="toasts" id="toasts"></div>
<div class="modal-bg" id="modal-bg" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <h3 id="modal-title">Details</h3>
    <div class="meta" id="modal-meta"></div>
    <pre id="modal-error"></pre>
    <button class="btn btn-ghost btn-sm" onclick="closeModal()">Close</button>
  </div>
</div>
<div class="modal-bg" id="move-bg" onclick="if(event.target===this)closeMove()">
  <div class="modal">
    <h3 style="color:#2563eb">Move document</h3>
    <div class="meta" id="move-meta"></div>
    <div class="field" style="margin-top:.6rem">
      <label>Target collection</label>
      <select id="move-target"></select>
    </div>
    <div style="display:flex;gap:.5rem;margin-top:1rem">
      <button class="btn btn-primary btn-sm" onclick="confirmMove()">Move</button>
      <button class="btn btn-ghost btn-sm" onclick="closeMove()">Cancel</button>
    </div>
  </div>
</div>
<div class="modal-bg" id="vendor-bg" onclick="if(event.target===this)closeVendor()">
  <div class="modal">
    <h3 style="color:#2563eb">Set vendor / source tag</h3>
    <div class="meta" id="vendor-meta"></div>
    <div class="field" style="margin-top:.6rem">
      <label>New vendor / source tag</label>
      <input type="text" id="vendor-input" placeholder="e.g. HPE, Dell, engineering" onkeydown="if(event.key==='Enter')confirmVendor()">
    </div>
    <div style="display:flex;gap:.5rem;margin-top:1rem">
      <button class="btn btn-primary btn-sm" onclick="confirmVendor()">Apply</button>
      <button class="btn btn-ghost btn-sm" onclick="closeVendor()">Cancel</button>
    </div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);

function toast(msg, type='info') {
  const el = document.createElement('div');
  el.className = 'toast ' + type;
  el.textContent = msg;
  $('toasts').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function col() { return $('col-sel').value; }
function vendor() { return $('vendor').value.trim() || 'admin'; }

// ── collections ──────────────────────────────────────────────────────────────
let collectionsCache = [];
async function loadCollections() {
  const data = await fetch('/collections').then(r=>r.json()).catch(()=>({collections:[]}));
  const cols = (data.collections || []).map(c => c.name);
  collectionsCache = cols;
  ['col-sel','filter-col'].forEach(id => {
    const sel = $(id);
    const placeholder = id === 'filter-col'
      ? '<option value="">All collections</option>'
      : '<option value="">Select collection…</option>';
    sel.innerHTML = placeholder + cols.map(n => '<option value="'+esc(n)+'">'+esc(n)+'</option>').join('');
  });
}

function toggleNewCol() {
  const row = $('new-col-row');
  row.style.display = row.style.display === 'none' ? '' : 'none';
}

async function createCollection() {
  const name = $('new-col-input').value.trim();
  if (!name) return;
  const r = await fetch('/collections', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({name})
  });
  if (r.ok) {
    toast('Collection "' + name + '" created', 'success');
    $('new-col-input').value = '';
    $('new-col-row').style.display = 'none';
    await loadCollections();
    $('col-sel').value = name;
  } else {
    toast('Failed to create collection', 'error');
  }
}

// ── documents ─────────────────────────────────────────────────────────────────
let docsCache = [];
let sortKey = 'updated_at';
let sortDir = -1;  // 1 = ascending, -1 = descending

const TYPE_ICON = {url:'🔗', document:'📄', deep_crawl:'🕷️'};
const STATUS_LABEL = {completed:'done', failed:'error', unchanged:'unchanged'};

function srcOf(d) { return d.url || d.id || ''; }

// ── selection / bulk actions (Phase 7.8) ──────────────────────────────────────
let selected = new Set();   // doc ids checked (persists across sort/filter/refresh)
let lastIds = [];           // ids currently rendered (post-filter), for select-all
let bulkBusy = false;       // pause auto-refresh while a bulk op runs

function updateBulkBar() {
  const n = selected.size;
  $('bulk-bar').style.display = n ? 'flex' : 'none';
  $('bulk-count').innerHTML = '<b>' + n + '</b> selected';
  const sa = $('select-all');
  const vis = lastIds.filter(id => selected.has(id)).length;
  sa.checked = lastIds.length > 0 && vis === lastIds.length;
  sa.indeterminate = vis > 0 && vis < lastIds.length;
}
function clearSelection() { selected.clear(); renderDocs(); }

async function loadDocs() {
  if (bulkBusy) return;  // don't yank the table out from under a running bulk op
  const c = $('filter-col').value;
  const url = '/documents' + (c ? '?collection=' + encodeURIComponent(c) : '');
  const data = await fetch(url).then(r=>r.json()).catch(()=>({documents:[]}));
  docsCache = data.documents || [];
  const ids = new Set(docsCache.map(d => d.id));
  selected.forEach(id => { if (!ids.has(id)) selected.delete(id); });  // drop gone docs
  renderDocs();
  syncQueueBadges();
}

function sortBy(k) {
  if (sortKey === k) sortDir = -sortDir;
  else { sortKey = k; sortDir = 1; }
  renderDocs();
}

function renderDocs() {
  const tbody = $('docs-body');
  const q = $('filter-text').value.trim().toLowerCase();

  let rows = docsCache.slice();
  if (q) {
    rows = rows.filter(d =>
      (srcOf(d) + ' ' + (d.collection||'') + ' ' + (d.vendor||'') + ' ' +
       (d.source_type||'') + ' ' + (d.status||'')).toLowerCase().includes(q)
    );
  }

  rows.sort((a, b) => {
    let va = sortKey === 'src' ? srcOf(a) : a[sortKey];
    let vb = sortKey === 'src' ? srcOf(b) : b[sortKey];
    va = (va == null ? '' : String(va)).toLowerCase();
    vb = (vb == null ? '' : String(vb)).toLowerCase();
    if (va < vb) return -sortDir;
    if (va > vb) return sortDir;
    return 0;
  });

  document.querySelectorAll('th .arrow').forEach(s => {
    s.textContent = (s.dataset.k === sortKey) ? (sortDir === 1 ? '▲' : '▼') : '';
  });

  lastIds = rows.map(d => d.id);

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#94a3b8;padding:2rem">' +
      (docsCache.length ? 'No documents match the filter' : 'No documents yet') + '</td></tr>';
    updateBulkBar();
    return;
  }

  tbody.innerHTML = rows.map(d => {
    const src = srcOf(d);
    const short = src.length > 55 ? '…' + src.slice(-52) : src;
    const date = (d.updated_at || '').slice(0, 16).replace('T', ' ') || '—';
    const icon = TYPE_ICON[d.source_type] || '📄';
    const statusLabel = STATUS_LABEL[d.status] || d.status;
    const isErr = d.status === 'failed';
    const badge = '<span class="sb s-' + esc(d.status) + (isErr ? ' clickable" data-err="' + esc(d.id) : '') +
      '"' + (isErr ? ' title="Click for the failure reason"' : '') + '>' + esc(statusLabel) + '</span>';
    return '<tr>' +
      '<td class="cbcol"><input type="checkbox" class="rowcb" data-id="' + esc(d.id) + '"' + (selected.has(d.id) ? ' checked' : '') + '></td>' +
      '<td class="src" title="' + esc(src) + '">' + esc(short) + '</td>' +
      '<td>' + esc(d.collection || '') + '</td>' +
      '<td>' + esc(d.vendor || '—') + '</td>' +
      '<td>' + icon + ' ' + esc(d.source_type || '') + '</td>' +
      '<td>' + badge + '</td>' +
      '<td style="white-space:nowrap;color:#64748b">' + date + '</td>' +
      '<td style="white-space:nowrap"><button class="btn btn-ghost btn-sm" data-move="' + esc(d.id) + '" title="Move to another collection">&#8631;</button> ' +
      '<button class="btn btn-danger btn-sm" data-del="' + esc(d.id) + '">&#x2715;</button></td>' +
      '</tr>';
  }).join('');
  updateBulkBar();
}

// delegated row actions — robust to ids (no inline-onclick string building)
$('docs-body').addEventListener('click', e => {
  const del = e.target.closest('[data-del]');
  if (del) { deleteDoc(del.dataset.del); return; }
  const mv = e.target.closest('[data-move]');
  if (mv) { showMove(mv.dataset.move); return; }
  const err = e.target.closest('[data-err]');
  if (err) { showError(err.dataset.err); }
});

// row checkbox + select-all (filtered) drive the selection set
$('docs-body').addEventListener('change', e => {
  const cb = e.target.closest('.rowcb');
  if (!cb) return;
  if (cb.checked) selected.add(cb.dataset.id); else selected.delete(cb.dataset.id);
  updateBulkBar();
});
$('select-all').addEventListener('change', e => {
  if (e.target.checked) lastIds.forEach(id => selected.add(id));
  else lastIds.forEach(id => selected.delete(id));
  renderDocs();
});

async function deleteDoc(id) {
  if (!confirm('Delete this document and remove its vectors from Qdrant?')) return;
  const r = await fetch('/documents/' + encodeURIComponent(id), {method:'DELETE'});
  toast(r.ok ? 'Document deleted' : 'Delete failed', r.ok ? 'info' : 'error');
  loadDocs();
}

function showError(id) {
  const d = docsCache.find(x => x.id === id);
  if (!d) return;
  $('modal-title').textContent = 'Ingestion failed';
  $('modal-meta').innerHTML =
    '<div><b>Source:</b> ' + esc(srcOf(d)) + '</div>' +
    '<div><b>Collection:</b> ' + esc(d.collection || '—') + ' &nbsp; <b>Vendor:</b> ' + esc(d.vendor || '—') + '</div>' +
    '<div><b>Updated:</b> ' + esc((d.updated_at || '').replace('T', ' ')) + '</div>';
  $('modal-error').textContent = d.error || '(no error detail was recorded)';
  $('modal-bg').classList.add('show');
}

function closeModal() { $('modal-bg').classList.remove('show'); }

// ── move document (Phase 7.P4) ────────────────────────────────────────────────
let moveDocId = null;
function showMove(id) {
  const d = docsCache.find(x => x.id === id);
  if (!d) return;
  moveDocId = id;
  $('move-meta').innerHTML =
    '<div><b>Document:</b> ' + esc(srcOf(d)) + '</div>' +
    '<div><b>Current collection:</b> ' + esc(d.collection || '—') + '</div>';
  const others = collectionsCache.filter(n => n !== d.collection);
  $('move-target').innerHTML = others.length
    ? others.map(n => '<option value="'+esc(n)+'">'+esc(n)+'</option>').join('')
    : '<option value="">(create another collection first)</option>';
  $('move-bg').classList.add('show');
}
function closeMove() { $('move-bg').classList.remove('show'); moveDocId = null; }
function moveOne(id, target) {
  return fetch('/documents/' + encodeURIComponent(id) + '/move', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({target_collection: target})
  });
}
async function confirmMove() {
  const target = $('move-target').value;
  if (!target) { toast('Pick a target collection', 'error'); return; }
  const single = moveDocId;          // null => bulk mode (use the selection)
  const ids = single ? [single] : [...selected];
  closeMove();
  if (single) {
    const r = await moveOne(single, target);
    if (r.ok) { toast('Document moved to "' + target + '"', 'success'); loadDocs(); }
    else { const d = await r.json().catch(()=>({})); toast('Move failed: ' + (d.detail || r.status), 'error'); }
  } else {
    await runBulk(ids, id => moveOne(id, target), 'Moved to "' + target + '":');
  }
}

// ── bulk actions over the current selection (Phase 7.8) ───────────────────────
// Reuses the validated single-doc endpoints; iterates sequentially so the
// server isn't hammered and partial failures are reported, not silent.
async function runBulk(ids, fn, verb) {
  if (!ids.length) return;
  bulkBusy = true;
  let ok = 0, fail = 0;
  for (const id of ids) {
    try { (await fn(id)).ok ? ok++ : fail++; } catch { fail++; }
  }
  bulkBusy = false;
  selected.clear();
  toast(verb + ' ' + ok + ' of ' + ids.length + (fail ? ' (' + fail + ' failed)' : ''), fail ? 'error' : 'success');
  loadDocs();
}
async function bulkDelete() {
  const ids = [...selected];
  if (!ids.length) return;
  if (!confirm('Delete ' + ids.length + ' document(s) and remove their vectors from Qdrant?')) return;
  await runBulk(ids, id => fetch('/documents/' + encodeURIComponent(id), {method:'DELETE'}), 'Deleted');
}
function showBulkMove() {
  if (!selected.size) return;
  moveDocId = null;  // bulk mode
  $('move-meta').innerHTML = '<div><b>' + selected.size + ' document(s)</b> selected</div>';
  $('move-target').innerHTML = collectionsCache.length
    ? collectionsCache.map(n => '<option value="'+esc(n)+'">'+esc(n)+'</option>').join('')
    : '<option value="">(create a collection first)</option>';
  $('move-bg').classList.add('show');
}

// ── set vendor / source tag over the current selection ────────────────────────
function showBulkVendor() {
  if (!selected.size) return;
  $('vendor-meta').innerHTML = '<div><b>' + selected.size + ' document(s)</b> selected</div>';
  $('vendor-input').value = '';
  $('vendor-bg').classList.add('show');
  $('vendor-input').focus();
}
function closeVendor() { $('vendor-bg').classList.remove('show'); }
function setVendorOne(id, v) {
  return fetch('/documents/' + encodeURIComponent(id) + '/vendor', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({vendor: v})
  });
}
async function confirmVendor() {
  const v = $('vendor-input').value.trim();
  if (!v) { toast('Enter a vendor / source tag', 'error'); return; }
  const ids = [...selected];
  closeVendor();
  await runBulk(ids, id => setVendorOne(id, v), 'Re-tagged');
}

// ── rename collection (Phase 7.P5) ────────────────────────────────────────────
function toggleRenameCol() {
  const cur = col();
  if (!cur) { toast('Select a collection to rename first', 'error'); return; }
  const row = $('rename-col-row');
  const showing = row.style.display !== 'none';
  row.style.display = showing ? 'none' : '';
  if (!showing) $('rename-col-input').value = cur;
}
async function renameCollection() {
  const cur = col();
  const newName = $('rename-col-input').value.trim();
  if (!cur || !newName || newName === cur) return;
  if (!confirm('Rename collection "' + cur + '" to "' + newName + '"?\nThis moves all its documents and vectors.')) return;
  const r = await fetch('/collections/' + encodeURIComponent(cur) + '/rename', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({new_name: newName})
  });
  if (r.ok) {
    toast('Renamed to "' + newName + '"', 'success');
    $('rename-col-row').style.display = 'none';
    await loadCollections();
    $('col-sel').value = newName;
    loadDocs();
  } else {
    const d = await r.json().catch(()=>({}));
    toast('Rename failed: ' + (d.detail || r.status), 'error');
  }
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeModal(); closeMove(); } });

// ── file upload ───────────────────────────────────────────────────────────────
const dz = $('dz');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('over'); handleFiles([...e.dataTransfer.files]); });
$('file-input').addEventListener('change', e => { handleFiles([...e.target.files]); e.target.value = ''; });

async function handleFiles(files) {
  if (!col()) { toast('Select a collection first', 'error'); return; }
  for (const f of files) await uploadFile(f);
  setTimeout(loadDocs, 1000);
}

async function uploadFile(file) {
  const item = document.createElement('div');
  item.className = 'qi';
  item.innerHTML = '<span class="qname">' + esc(file.name) + '</span><span class="sb s-uploading">uploading…</span>';
  $('queue').prepend(item);

  const fd = new FormData();
  fd.append('file', file);
  fd.append('collection', col());
  fd.append('vendor', vendor());

  try {
    const r = await fetch('/ingest/document', {method:'POST', body:fd});
    const data = await r.json();
    const badge = item.querySelector('.sb');
    if (r.ok) {
      badge.className = 'sb s-pending'; badge.textContent = 'queued';
      if (data.doc_id) item.dataset.docid = data.doc_id;  // tracked until a terminal status
      toast(file.name + ' queued', 'success');
    } else {
      badge.className = 'sb s-failed'; badge.textContent = 'error';
      toast(file.name + ': ' + (data.detail || 'error'), 'error');
      setTimeout(() => item.remove(), 8000);
    }
  } catch {
    const badge = item.querySelector('.sb');
    badge.className = 'sb s-failed'; badge.textContent = 'failed';
    toast('Upload failed: ' + file.name, 'error');
    setTimeout(() => item.remove(), 8000);
  }
}

// reflect the real ingestion outcome on queued upload items, then retire them
function syncQueueBadges() {
  document.querySelectorAll('#queue .qi[data-docid]').forEach(item => {
    const d = docsCache.find(x => x.id === item.dataset.docid);
    if (!d || !['completed','failed','unchanged'].includes(d.status)) return;
    const badge = item.querySelector('.sb');
    badge.className = 'sb s-' + d.status;
    badge.textContent = STATUS_LABEL[d.status] || d.status;
    delete item.dataset.docid;
    setTimeout(() => item.remove(), 6000);
  });
}

// ── URL ingestion ─────────────────────────────────────────────────────────────
async function ingestUrl() {
  const url = $('url-in').value.trim();
  if (!url) return;
  if (!col()) { toast('Select a collection first', 'error'); return; }
  const r = await fetch('/ingest/url', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({url, collection:col(), vendor:vendor()})
  });
  if (r.ok) {
    toast('URL queued: ' + url, 'success');
    $('url-in').value = '';
  } else {
    const d = await r.json();
    toast('Error: ' + (d.detail || r.status), 'error');
  }
  setTimeout(loadDocs, 1000);
}

async function deepCrawl() {
  const url = $('url-in').value.trim();
  if (!url) { toast('Enter a URL above first', 'error'); return; }
  if (!col()) { toast('Select a collection first', 'error'); return; }
  const payload = {
    url, collection:col(), vendor:vendor(),
    max_depth: parseInt($('depth').value) || 2,
    max_pages: parseInt($('pages').value) || 30,
  };
  const pat = $('pattern').value.trim();
  if (pat) payload.include_pattern = pat;
  const r = await fetch('/ingest/deep', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  });
  if (r.ok) {
    toast('Deep crawl queued: ' + url, 'success');
    $('url-in').value = '';
  } else {
    const d = await r.json();
    toast('Error: ' + (d.detail || r.status), 'error');
  }
  setTimeout(loadDocs, 1000);
}

// ── init ──────────────────────────────────────────────────────────────────────
loadCollections();
loadDocs();
setInterval(loadDocs, 10000);
</script>
</body>
</html>
"""


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index():
    if AUTH_ENABLED:
        badge = "Internal tool &middot; Basic Auth &middot; LAN only"
        note = ("&#128274; <strong>Basic Auth enabled.</strong> Access requires the admin credentials. "
                "Keep network access LAN-scoped as well. See <code>docs/ENHANCEMENT-PLAN.md</code>.")
    else:
        badge = "Internal tool &middot; no auth &middot; LAN only"
        note = ("&#9888; <strong>No authentication.</strong> This page is accessible to anyone on the LAN. "
                "Enable HTTP Basic Auth by setting <code>ADMIN_USER</code> / <code>ADMIN_PASSWORD</code>. "
                "See <code>docs/ENHANCEMENT-PLAN.md</code>.")
    return HTML.replace("<!--AUTH_BADGE-->", badge).replace("<!--AUTH_NOTE-->", note)


@app.get("/collections")
async def get_collections():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{INGESTION_URL}/collections")
        return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.post("/collections")
async def create_collection(request: Request):
    body = await request.json()
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(f"{INGESTION_URL}/collections", json=body)
        return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.post("/collections/{name}/rename")
async def rename_collection(name: str, request: Request):
    body = await request.json()
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(f"{INGESTION_URL}/collections/{quote(name, safe='')}/rename", json=body)
        return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.get("/documents")
async def get_documents(request: Request):
    params = dict(request.query_params)
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{INGESTION_URL}/documents", params=params)
        return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.delete(f"{INGESTION_URL}/documents/{doc_id}")
        return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.post("/documents/{doc_id}/move")
async def move_document(doc_id: str, request: Request):
    body = await request.json()
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{INGESTION_URL}/documents/{doc_id}/move", json=body)
        return Response(content=r.content, status_code=r.status_code, media_type="application/json")

@app.post("/documents/{doc_id}/vendor")
async def set_document_vendor(doc_id: str, request: Request):
    body = await request.json()
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{INGESTION_URL}/documents/{doc_id}/vendor", json=body)
        return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.post("/ingest/url")
async def ingest_url(request: Request):
    body = await request.json()
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{INGESTION_URL}/ingest/url", json=body)
        return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.post("/ingest/deep")
async def ingest_deep(request: Request):
    body = await request.json()
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{INGESTION_URL}/ingest/deep", json=body)
        return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.post("/ingest/document")
async def ingest_document(
    file: UploadFile = File(...),
    collection: str = Form(...),
    vendor: str = Form(...),
    access_roles: str = Form(default="all"),
    classification: str = Form(default="public"),
):
    content = await file.read()
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{INGESTION_URL}/ingest/document",
            files={"file": (file.filename, content, file.content_type or "application/octet-stream")},
            data={
                "collection": collection,
                "vendor": vendor,
                "access_roles": access_roles,
                "classification": classification,
            },
        )
        return Response(content=r.content, status_code=r.status_code, media_type="application/json")
