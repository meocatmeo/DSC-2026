"""Chunk store: one SQLite file holding documents and article level chunks.

Built from the supplied `context_*.json` files only. The store is the single
source of truth for both the lexical index and the dense index, so a chunk id
means the same thing everywhere.
"""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from .structure import document_chunks

SCHEMA = """
PRAGMA journal_mode=OFF;
PRAGMA synchronous=OFF;
CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT, number TEXT, subject TEXT,
                        chunks INTEGER, position INTEGER);
CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id TEXT NOT NULL, ord INTEGER,
                     unit TEXT, crumb TEXT, body TEXT);
CREATE INDEX chunks_document ON chunks(document_id, ord);
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_object)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_records(corpus):
    """Yield (name, raw bytes) for each supplied document.

    Accepts the organiser's `selected-contexts` directory of `context_*.json` files,
    or a single JSON Lines file (optionally gzipped) holding the same records - the
    latter makes it possible to move the corpus as one file without changing the
    parsing path.
    """
    corpus = Path(corpus)
    if corpus.is_dir():
        files = sorted(corpus.glob("context_*.json"))
        if not files:
            raise ValueError(f"No context_*.json files in {corpus}")
        for path in files:
            yield path.name, path.read_bytes()
        return
    if not corpus.is_file():
        raise ValueError(f"Corpus not found: {corpus}")
    opener = __import__("gzip").open if corpus.name.endswith(".gz") else open
    with opener(corpus, "rb") as stream:
        for number, line in enumerate(stream):
            line = line.strip()
            if line:
                yield f"line_{number}", line


def build(corpus, target, size=320, overlap=64, header_words=220):
    files = list(source_records(corpus))
    if not files:
        raise ValueError(f"No documents found in {corpus}")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.with_suffix(".building.sqlite")
    building.unlink(missing_ok=True)
    db = sqlite3.connect(building)
    db.executescript(SCHEMA)
    digest = hashlib.sha256()
    started, rowid, empty = time.time(), 0, []
    try:
        for position, (name, raw) in enumerate(files):
            digest.update(name.encode() + b"\0" + raw + b"\0")
            record = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
            doc_id = str(record["id"])
            pieces = document_chunks(record.get("name", ""), record.get("passage", ""),
                                     size=size, overlap=overlap, header_words=header_words)
            if not record.get("passage"):
                empty.append(doc_id)
            head = pieces[0]
            db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?)",
                       (doc_id, head["title"], head["number"], head["subject"], len(pieces), position))
            for order, chunk in enumerate(pieces):
                rowid += 1
                db.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                           (rowid, doc_id, order, chunk["unit"], chunk["crumb"], chunk["body"]))
            if (position + 1) % 1000 == 0:
                print(f"  {position + 1}/{len(files)} documents, {rowid} chunks, "
                      f"{time.time() - started:.0f}s", flush=True)
        report = {"documents": len(files), "chunks": rowid, "empty_passage_ids": empty,
                  "corpus_sha256": digest.hexdigest(), "chunk_words": size,
                  "overlap_words": overlap, "header_words": header_words, "format_version": 2}
        db.execute("INSERT INTO metadata VALUES ('corpus', ?)", (json.dumps(report),))
        db.commit()
    finally:
        db.close()
    building.replace(target)
    write_json(target.with_suffix(".json"), report)
    print(f"Built {rowid} chunks from {len(files)} documents in {time.time() - started:.0f}s")
    return report


def connect(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Build the chunk store first: {path}")
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


class Store:
    """Read only access to documents, chunks and their surrounding context."""

    def __init__(self, path):
        self.path = Path(path)
        self.db = connect(path)
        self.metadata = json.loads(
            self.db.execute("SELECT value FROM metadata WHERE key='corpus'").fetchone()[0])
        rows = self.db.execute("SELECT id,title,number,subject FROM documents ORDER BY id").fetchall()
        self.doc_ids = [r[0] for r in rows]
        # Iterate the id list itself: enumerating rows-vs-ids here is easy to get wrong,
        # and a wrong mapping makes every `g in store.doc_position` check silently false,
        # which mines zero training groups instead of raising.
        self.doc_position = {doc: i for i, doc in enumerate(self.doc_ids)}
        if len(self.doc_position) != len(self.doc_ids):
            raise ValueError("Duplicate document ids in the chunk store")
        self.labels = {r[0]: " | ".join(x for x in (r[1], r[2], r[3]) if x) for r in rows}
        self._text = {}

    def label(self, doc_id):
        return self.labels.get(doc_id, "")

    def chunk(self, chunk_id):
        if chunk_id not in self._text:
            row = self.db.execute(
                "SELECT document_id, ord, unit, crumb, body FROM chunks WHERE id=?",
                (chunk_id,)).fetchone()
            self._text[chunk_id] = row
        return self._text[chunk_id]

    def chunk_text(self, chunk_id, with_label=True):
        document_id, _, _, crumb, body = self.chunk(chunk_id)
        head = " | ".join(x for x in ((self.label(document_id) if with_label else ""), crumb) if x)
        return (head + "\n" + body).strip() if head else body

    def document_chunk_ids(self, doc_id):
        return [r[0] for r in self.db.execute(
            "SELECT id FROM chunks WHERE document_id=? ORDER BY ord", (doc_id,))]

    def chunk_documents(self):
        return dict(self.db.execute("SELECT id, document_id FROM chunks"))

    def close(self):
        self.db.close()
