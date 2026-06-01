"""Ingest the reference corpus into the main knowledge store.

Usage (offline / file store, recommended subset for the MBBS beta):

    python -m app.ingestion.ingest \
        --corpus "/Users/naveen/Downloads/RAG DB" \
        --out data/ent_index.npz \
        --provider hashing --dim 384 \
        --books dhingra bansal

For the real test, switch ``--provider sentence_transformers`` (or ``openai``).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from typing import Dict, List, Optional

from ..providers.embeddings import build_embedder
from ..retrieval.main_store import Chunk, FileMainStore
from .chunker import chunk_pages
from .pdf_loader import load_pdf_pages, pdf_page_count

# Human-readable, citable source names for the known corpus. MBBS-tier texts are
# marked so a student-facing run can prioritise them.
SOURCE_NAMES: Dict[str, Dict[str, object]] = {
    "dhingra": {"name": "Dhingra — Diseases of Ear, Nose and Throat", "tier": "mbbs"},
    "bansal": {"name": "Bansal — Diseases of Ear, Nose and Throat", "tier": "mbbs"},
    "levine": {"name": "Levine — Rhinology", "tier": "specialist"},
    "wormwald": {"name": "Wormald — Endoscopic Sinus Surgery", "tier": "specialist"},
    "wormald": {"name": "Wormald — Endoscopic Sinus Surgery", "tier": "specialist"},
    "ballenger": {"name": "Ballenger's Otorhinolaryngology Head & Neck Surgery", "tier": "specialist"},
    "cummings": {"name": "Cummings Otolaryngology (6th ed.)", "tier": "specialist"},
    "scott brown vol1": {"name": "Scott-Brown's Otorhinolaryngology Vol 1 (Rhinology)", "tier": "specialist"},
    "scott brown vol2": {"name": "Scott-Brown's Vol 2 (Paediatrics, Ear, Skull Base)", "tier": "specialist"},
    "scott brown vol3": {"name": "Scott-Brown's Vol 3 (Head & Neck Oncology)", "tier": "specialist"},
    "tumors of the nose": {"name": "Lund — Tumors of the Nose, Sinuses, and Nasopharynx", "tier": "specialist"},
    "lund tumors": {"name": "Lund — Tumors of the Nose, Sinuses, and Nasopharynx", "tier": "specialist"},
}


def _slug(filename: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", os.path.splitext(filename.lower())[0]).strip()


def source_for(filename: str) -> Dict[str, object]:
    s = _slug(filename)
    for key, meta in SOURCE_NAMES.items():
        if key in s:
            return meta
    return {"name": os.path.splitext(filename)[0], "tier": "unknown"}


def _file_md5(path: str, limit_bytes: int = 8_000_000) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(limit_bytes))
    return h.hexdigest()


def discover_pdfs(corpus_dir: str, books: Optional[List[str]]) -> List[str]:
    files = sorted(f for f in os.listdir(corpus_dir) if f.lower().endswith(".pdf"))
    if books:
        wanted = [b.lower() for b in books]
        files = [f for f in files if any(w in f.lower() for w in wanted)]
    return [os.path.join(corpus_dir, f) for f in files]


def ingest(
    corpus_dir: str, out_path: str, *, provider: str, dim: int, model: str = "",
    books: Optional[List[str]] = None, max_pages: Optional[int] = None,
    chunk_chars: int = 1200, overlap: int = 200,
) -> Dict[str, object]:
    embedder = build_embedder(provider, dim=dim, model=model)
    store = FileMainStore(out_path, dim=embedder.dim, embedder_name=embedder.name)
    pdfs = discover_pdfs(corpus_dir, books)
    if not pdfs:
        raise SystemExit(f"No matching PDFs in {corpus_dir!r}")

    seen_hashes: set[str] = set()
    stats = {"books": [], "total_chunks": 0, "skipped_duplicates": []}

    for path in pdfs:
        fname = os.path.basename(path)
        digest = _file_md5(path)
        if digest in seen_hashes:
            print(f"  ↪ skipping duplicate: {fname}")
            stats["skipped_duplicates"].append(fname)
            continue
        seen_hashes.add(digest)

        meta = source_for(fname)
        source_name = str(meta["name"])
        slug = _slug(fname).replace(" ", "_")
        npages = pdf_page_count(path)
        print(f"• {source_name}  ({npages} pages){'  [cap %d]' % max_pages if max_pages else ''}")

        t0 = time.time()
        pages = load_pdf_pages(path, max_pages=max_pages)
        specs = chunk_pages(pages, chunk_chars=chunk_chars, overlap=overlap)
        if not specs:
            print("  ⚠ no extractable text (scanned? needs OCR) — skipped")
            continue

        embeddings = embedder.embed_batch([s.content for s in specs])
        chunks = [
            Chunk(
                id=f"{slug}:{i}:{s.page_start}-{s.page_end}",
                content=s.content, source=source_name,
                page_start=s.page_start, page_end=s.page_end, section=str(meta.get("tier", "")),
            )
            for i, s in enumerate(specs)
        ]
        store.add(chunks, embeddings)
        dt = time.time() - t0
        stats["total_chunks"] += len(chunks)
        stats["books"].append({"source": source_name, "chunks": len(chunks), "seconds": round(dt, 1)})
        print(f"  ✓ {len(chunks)} chunks in {dt:.1f}s")

    store.save()
    print(f"\n✅ Ingested {stats['total_chunks']} chunks from {len(stats['books'])} book(s) "
          f"using '{embedder.name}' (dim={embedder.dim}) → {out_path}")
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Ingest ENT reference corpus into the main store.")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", default="data/ent_index.npz")
    p.add_argument("--provider", default="hashing", choices=["hashing", "sentence_transformers", "openai"])
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--model", default="")
    p.add_argument("--books", nargs="*", default=None, help="substring filter, e.g. dhingra bansal")
    p.add_argument("--max-pages", type=int, default=None, help="cap pages per book (testing)")
    p.add_argument("--chunk-chars", type=int, default=1200)
    p.add_argument("--overlap", type=int, default=200)
    args = p.parse_args(argv)
    ingest(args.corpus, args.out, provider=args.provider, dim=args.dim, model=args.model,
           books=args.books, max_pages=args.max_pages,
           chunk_chars=args.chunk_chars, overlap=args.overlap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
