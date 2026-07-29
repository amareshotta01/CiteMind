"""
Quick manual test — not part of the app, just to verify loaders.py +
chunker.py work correctly together before wiring them into the API.
Run with: python -m backend.ingest.test_manual
"""

from backend.ingest.loaders import load_document
from backend.ingest.chunker import chunk_pages

with open("sample.pdf", "rb") as f:
    file_bytes = f.read()

doc = load_document("sample.pdf", file_bytes)
chunks = chunk_pages(doc_id="test-doc-1", pages=doc["content"])

print(f"Loaded {len(doc['content'])} pages")
print(f"Produced {len(chunks)} chunks")
print("\nAll chunks:")
for c in chunks:
    print(f"  page {c['page']}, offset {c['char_offset']}: {c['text'][:60]}...")