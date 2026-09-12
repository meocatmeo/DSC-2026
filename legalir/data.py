"""Strict JSON, corpus preparation, and deterministic holdouts (stdlib only)."""

import hashlib
import json
import random
import re
import sqlite3
import unicodedata
from pathlib import Path


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
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def terms(text):
    return re.findall(r"\w+", normalize(text).lower(), flags=re.UNICODE)


def chunks(text, size=180, overlap=40):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require chunk size > overlap >= 0")
    words = normalize(text).split()
    for start in range(0, len(words), size - overlap):
        yield " ".join(words[start:start + size])
        if start + size >= len(words):
            break


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def connect(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Prepare the corpus first: {path}")
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def prepare(corpus, target, size=180, overlap=40):
    # Validate before touching output. Never follow source URLs.
    list(chunks("check", size, overlap))
    files = sorted(Path(corpus).glob("context_*.json"))
    if not files:
        raise ValueError(f"No context files in {corpus}")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".building.sqlite")
    temporary.unlink(missing_ok=True)
    db = sqlite3.connect(temporary)
    digest = hashlib.sha256()
    empty_ids = []
    try:
        db.executescript("""
            CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT NOT NULL);
            CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id TEXT NOT NULL,
                                 text TEXT NOT NULL);
            CREATE INDEX chunks_document ON chunks(document_id);
            CREATE VIRTUAL TABLE search USING fts5(text, content='chunks',
                content_rowid='id', tokenize='unicode61 remove_diacritics 0');
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        for number, path in enumerate(files, 1):
            raw = path.read_bytes()
            digest.update(path.name.encode() + b"\0" + raw + b"\0")
            record = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
            doc_id = str(record["id"])
            title = normalize(record.get("name", "").replace("-", " "))
            passage = normalize(record.get("passage", ""))
            db.execute("INSERT INTO documents VALUES (?, ?)", (doc_id, title))
            if not passage:
                empty_ids.append(doc_id)
            passages = list(chunks(passage, size, overlap)) or [""]
            for passage_chunk in passages:
                text = normalize(title + "\n" + passage_chunk)
                db.execute("INSERT INTO chunks(document_id,text) VALUES (?,?)", (doc_id, text))
            if number % 1000 == 0:
                print(f"Prepared {number}/{len(files)} documents", flush=True)
        db.execute("INSERT INTO search(search) VALUES ('rebuild')")
        report = {"documents": len(files), "chunks": db.execute("SELECT count(*) FROM chunks").fetchone()[0],
                  "empty_passage_ids": empty_ids, "corpus_sha256": digest.hexdigest(),
                  "chunk_words": size, "overlap_words": overlap, "format_version": 1}
        db.execute("INSERT INTO metadata VALUES ('corpus', ?)", (json.dumps(report),))
        db.commit()
    finally:
        db.close()
    temporary.replace(target)
    write_json(target.with_suffix(".json"), report)
    return report


def load_questions(path, labeled=False):
    records = read_json(path)
    if not isinstance(records, dict) or not records:
        raise ValueError("Questions must be a nonempty JSON object")
    for key, value in records.items():
        if not isinstance(value, dict) or not isinstance(value.get("question"), str):
            raise ValueError(f"Invalid question record: {key}")
        if labeled:
            answers = value.get("answer")
            if not isinstance(answers, list) or not answers or any(not isinstance(x, str) for x in answers):
                raise ValueError(f"Invalid training answers: {key}")
            if len(answers) != len(set(answers)):
                raise ValueError(f"Duplicate training answers: {key}")
    return records


def split_questions(source, directory, fraction=0.15, seed=2026):
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    records = load_questions(source, labeled=True)
    # Exact normalized duplicate questions stay together. This is not augmentation.
    groups = {}
    for key, record in sorted(records.items()):
        groups.setdefault(normalize(record["question"]).casefold(), []).append(key)
    names = sorted(groups)
    if len(names) < 2:
        raise ValueError("Need at least two distinct question groups")
    random.Random(seed).shuffle(names)
    target = max(1, round(len(records) * fraction))
    validation_ids = set()
    for name in names[:-1]:
        validation_ids.update(groups[name])
        if len(validation_ids) >= target:
            break
    train = {k: v for k, v in records.items() if k not in validation_ids}
    validation = {k: v for k, v in records.items() if k in validation_ids}
    directory = Path(directory)
    write_json(directory / "train.json", train)
    write_json(directory / "validation.json", validation)
    report = {"seed": seed, "source_sha256": file_hash(source), "train": len(train),
              "validation": len(validation), "grouping": "NFC, whitespace, casefold exact question"}
    write_json(directory / "split.json", report)
    return report