#!/usr/bin/env python3
"""One-time migration: copy already-ingested originals from the legacy FILES_DIR into the
DOC_STORE as <vendor>/<filename>, and record file_path in the DB — so documents ingested
before the document store existed become browsable on the share and get a working
/documents/{id}/file link. Idempotent: skips docs already migrated.

Run inside the ingestion container (it has FILES_DIR / DOC_STORE / DB_PATH and the store mount):
    docker compose exec -T ingestion python - < scripts/migrate-doc-store.py
"""
import glob
import os
import sqlite3

DB_PATH   = os.getenv("DB_PATH",   "/app/data/ingestion.db")
FILES_DIR = os.getenv("FILES_DIR", "/app/data/files")
DOC_STORE = os.getenv("DOC_STORE", "")

if not DOC_STORE:
    raise SystemExit("DOC_STORE is not set — nothing to migrate into.")


def _safe(name: str) -> str:
    return os.path.basename((name or "").strip()).lstrip(".")


conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT id, url, vendor, file_path FROM documents WHERE source_type='document'"
).fetchall()

migrated = skipped = missing = 0
for r in rows:
    doc_id, filename, vendor, file_path = r["id"], r["url"], r["vendor"], r["file_path"]
    if file_path and os.path.exists(os.path.join(DOC_STORE, file_path)):
        skipped += 1
        continue
    matches = glob.glob(os.path.join(FILES_DIR, f"{doc_id}.*"))
    if not matches:
        missing += 1
        print(f"  MISSING source for {doc_id} ({filename})")
        continue
    with open(matches[0], "rb") as f:
        content = f.read()

    sv = _safe(vendor) or "unknown"
    sn = _safe(filename) or f"{doc_id}.bin"
    os.makedirs(os.path.join(DOC_STORE, sv), exist_ok=True)
    target = os.path.join(DOC_STORE, sv, sn)
    if os.path.exists(target):
        with open(target, "rb") as existing:
            if existing.read() != content:                 # same name, different file
                stem, ext = os.path.splitext(sn)
                sn = f"{stem}-{doc_id[:8]}{ext}"
                target = os.path.join(DOC_STORE, sv, sn)
    with open(target, "wb") as f:
        f.write(content)

    conn.execute("UPDATE documents SET file_path=? WHERE id=?", (f"{sv}/{sn}", doc_id))
    migrated += 1

conn.commit()
conn.close()
print(f"Migrated {migrated}, already-present {skipped}, missing-source {missing}.")
