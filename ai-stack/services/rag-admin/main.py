import os
import httpx
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response

INGESTION_URL = os.environ.get("INGESTION_URL", "http://ingestion:8002")

app = FastAPI(docs_url=None, redoc_url=None)

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
.s-done{background:#dcfce7;color:#166534}
.s-error{background:#fee2e2;color:#991b1b}
.s-uploading{background:#e0e7ff;color:#3730a3}
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
</style>
</head>
<body>
<nav>
  <h1>Open RAG Stack &mdash; Admin</h1>
  <span class="badge">Internal tool &middot; no auth &middot; LAN only</span>
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
        </div>
      </div>
      <div class="field" id="new-col-row" style="display:none">
        <label>New collection name</label>
        <div style="display:flex;gap:.5rem">
          <input type="text" id="new-col-input" placeholder="e.g. company-policies">
          <button class="btn btn-primary btn-sm" onclick="createCollection()">Create</button>
        </div>
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
          <div style="display:flex;gap:.5rem">
            <div class="field" style="flex:1">
              <label>Max depth</label>
              <input type="number" id="depth" value="2" min="1" max="5">
            </div>
            <div class="field" style="flex:1">
              <label>Max pages</label>
              <input type="number" id="pages" value="30" min="1" max="200">
            </div>
          </div>
          <div class="field">
            <label>URL pattern filter (optional)</label>
            <input type="text" id="pattern" placeholder="e.g. /docs/.*">
          </div>
          <button class="btn btn-ghost" style="width:100%" onclick="deepCrawl()">Start deep crawl from URL above</button>
        </div>
      </details>

      <div class="note">
        &#9888; <strong>No authentication.</strong> This page is accessible to anyone on the LAN.
        Future work: contributor roles (per-user write access), ingestion audit log.
        See <code>docs/ENHANCEMENT-PLAN.md</code>.
      </div>
    </div>
  </div>

  <!-- ── Right: document list ──────────────────────────────────────── -->
  <div class="card">
    <div class="docs-hd">
      <h2>Documents</h2>
      <div class="sp"></div>
      <select id="filter-col" onchange="loadDocs()">
        <option value="">All collections</option>
      </select>
      <button class="btn btn-ghost btn-sm" onclick="loadDocs()">&#8635; Refresh</button>
    </div>
    <table>
      <thead>
        <tr>
          <th>Source</th>
          <th>Collection</th>
          <th>Vendor</th>
          <th>Type</th>
          <th>Status</th>
          <th>Updated</th>
          <th></th>
        </tr>
      </thead>
      <tbody id="docs-body">
        <tr><td colspan="7" style="text-align:center;color:#94a3b8;padding:2rem">Loading&hellip;</td></tr>
      </tbody>
    </table>
  </div>
</main>
<div class="toasts" id="toasts"></div>
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
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function col() { return $('col-sel').value; }
function vendor() { return $('vendor').value.trim() || 'admin'; }

// ── collections ──────────────────────────────────────────────────────────────
async function loadCollections() {
  const data = await fetch('/collections').then(r=>r.json()).catch(()=>({collections:[]}));
  const cols = (data.collections || []).map(c => c.name);
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
async function loadDocs() {
  const c = $('filter-col').value;
  const url = '/documents' + (c ? '?collection=' + encodeURIComponent(c) : '');
  const data = await fetch(url).then(r=>r.json()).catch(()=>({documents:[]}));
  const docs = data.documents || [];
  const tbody = $('docs-body');
  if (!docs.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#94a3b8;padding:2rem">No documents yet</td></tr>';
    return;
  }
  const TYPE_ICON = {url:'🔗', document:'📄', deep_crawl:'🕷️'};
  tbody.innerHTML = docs.map(d => {
    const src = d.url || d.id || '';
    const short = src.length > 55 ? '…' + src.slice(-52) : src;
    const date = (d.updated_at || '').slice(0, 16).replace('T', ' ') || '—';
    const icon = TYPE_ICON[d.source_type] || '📄';
    return '<tr>' +
      '<td class="src" title="' + esc(src) + '">' + esc(short) + '</td>' +
      '<td>' + esc(d.collection || '') + '</td>' +
      '<td>' + esc(d.vendor || '—') + '</td>' +
      '<td>' + icon + ' ' + esc(d.source_type || '') + '</td>' +
      '<td><span class="sb s-' + esc(d.status) + '">' + esc(d.status) + '</span></td>' +
      '<td style="white-space:nowrap;color:#64748b">' + date + '</td>' +
      '<td><button class="btn btn-danger btn-sm" onclick="deleteDoc(\'' + esc(d.id) + '\')">&#x2715;</button></td>' +
      '</tr>';
  }).join('');
}

async function deleteDoc(id) {
  if (!confirm('Delete this document and remove its vectors from Qdrant?')) return;
  const r = await fetch('/documents/' + id, {method:'DELETE'});
  toast(r.ok ? 'Document deleted' : 'Delete failed', r.ok ? 'info' : 'error');
  loadDocs();
}

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
      toast(file.name + ' queued', 'success');
    } else {
      badge.className = 'sb s-error'; badge.textContent = 'error';
      toast(file.name + ': ' + (data.detail || 'error'), 'error');
    }
  } catch {
    const badge = item.querySelector('.sb');
    badge.className = 'sb s-error'; badge.textContent = 'failed';
    toast('Upload failed: ' + file.name, 'error');
  }
  setTimeout(() => item.remove(), 30000);
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
    return HTML


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
