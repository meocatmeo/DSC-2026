"""Dense chunk retrieval with a locally stored bi-encoder.

Chunk vectors are written to a float16 memory mapped matrix; similarity search is
a single matrix product on the chosen device, in blocks so memory stays bounded.
The index records the checkpoint provenance and the store hash it was built from,
so a stale index cannot be used silently.
"""
import time
from pathlib import Path

import numpy as np

from .models import INSTRUCTION, Local, needs_instruction
from .store import Store, file_hash, read_json, write_json


class Embedder:
    def __init__(self, path, device="cuda", pooling=None):
        self.local = Local(path, kind="encoder", device=device)
        self.pooling = pooling or self.local.pooling
        self.instruct = needs_instruction(path)
        self.dimension = int(self.local.model.config.hidden_size)
        self.parameters = self.local.parameters

    def encode(self, texts, batch_size=16, max_length=1024, query=False, dtype="float16"):
        torch = self.local.torch
        out = np.empty((len(texts), self.dimension), dtype=dtype)
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            if query and self.instruct:
                batch = [f"Instruct: {INSTRUCTION}\nQuery: {t}" for t in batch]
            inputs = self.local.tokenizer(batch, padding=True, truncation=True,
                                          max_length=max_length, return_tensors="pt")
            inputs = {k: v.to(self.local.device) for k, v in inputs.items()}
            with torch.inference_mode():
                hidden = self.local.model(**inputs).last_hidden_state
                if self.pooling == "cls":
                    vectors = hidden[:, 0]
                elif self.pooling == "mean":
                    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                else:
                    # Left padding is used for last-token pooling, so -1 is a real token.
                    vectors = hidden[:, -1]
                vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=1)
            out[start:start + len(batch)] = vectors.cpu().numpy().astype(dtype)
        return out

    def close(self):
        self.local.close()


def build(store_path, model_path, directory, device="cuda", batch_size=16, max_length=1024):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").unlink(missing_ok=True)   # completion marker
    store = Store(store_path)
    count = store.metadata["chunks"]
    embedder = Embedder(model_path, device=device)
    vectors = np.lib.format.open_memmap(directory / "vectors.npy", mode="w+", dtype="float16",
                                        shape=(count, embedder.dimension))
    ids, offset, started = [], 0, time.time()
    cursor = store.db.execute(
        "SELECT c.id, d.title, d.number, d.subject, c.crumb, c.body "
        "FROM chunks c JOIN documents d ON d.id = c.document_id ORDER BY c.id")
    try:
        while batch := cursor.fetchmany(batch_size * 8):
            texts = []
            for _, title, number, subject, crumb, body in batch:
                head = " | ".join(x for x in (title, number, subject, crumb) if x)
                texts.append((head + "\n" + body).strip() if head else body)
            block = embedder.encode(texts, batch_size, max_length)
            vectors[offset:offset + len(batch)] = block
            ids.extend(int(r[0]) for r in batch)
            offset += len(batch)
            if offset % (batch_size * 80) < batch_size * 8:
                rate = offset / max(1e-6, time.time() - started)
                print(f"  embedded {offset}/{count} chunks, {rate:.0f}/s, "
                      f"eta {(count - offset) / max(rate, 1e-6) / 60:.0f} min", flush=True)
        vectors.flush()
        np.save(directory / "chunk_ids.npy", np.asarray(ids, dtype="int64"))
        manifest = {"store_sha256": file_hash(store_path), "model": embedder.local.provenance,
                    "parameters": embedder.parameters, "dimension": embedder.dimension,
                    "chunks": count, "max_length": max_length, "pooling": embedder.pooling,
                    "seconds": round(time.time() - started)}
        write_json(directory / "manifest.json", manifest)
    finally:
        vectors._mmap.close()
        embedder.close()
        store.close()
    return manifest


class DenseIndex:
    """Cosine search over the chunk matrix.

    The whole matrix is moved onto the device when it fits (341k x 1024 float16 is
    about 0.7 GiB), which turns each query into one matrix product; otherwise it is
    streamed in blocks so a smaller card still works.
    """

    def __init__(self, directory, store_path, model_path, device="cuda", block=200_000,
                 resident_limit=2 << 30):
        directory = Path(directory)
        self.manifest = read_json(directory / "manifest.json")
        if self.manifest["store_sha256"] != file_hash(store_path):
            raise ValueError("Dense index was built from a different chunk store; rebuild it")
        self.embedder = Embedder(model_path, device=device)
        if self.embedder.local.provenance != self.manifest["model"]:
            raise ValueError("Embedding checkpoint differs from the one the index was built with")
        self.vectors = np.load(directory / "vectors.npy", mmap_mode="r")
        self.chunk_ids = np.load(directory / "chunk_ids.npy")
        if self.vectors.shape != (self.manifest["chunks"], self.manifest["dimension"]):
            raise ValueError("Dense vector matrix has unexpected shape")
        self.parameters = self.embedder.parameters
        self.block = block
        self.torch = self.embedder.local.torch
        self.device = self.embedder.local.device
        self.max_length = self.manifest["max_length"]
        self.resident = None
        size = self.vectors.nbytes
        if self.device.startswith("cuda") and size <= resident_limit:
            try:
                self.resident = self.torch.from_numpy(
                    np.ascontiguousarray(self.vectors)).to(self.device)
                print(f"  dense matrix resident on {self.device} ({size / 2**30:.2f} GiB)",
                      flush=True)
            except RuntimeError:
                self.resident = None    # not enough free VRAM; fall back to blocks

    def scores(self, query):
        vector = self.embedder.encode([query], 1, 256, query=True)[0]
        torch = self.torch
        needle = torch.from_numpy(vector.astype("float32")).to(self.device)
        if self.resident is not None:
            with torch.inference_mode():
                return (self.resident.float() @ needle).cpu().numpy()
        out = np.empty(len(self.chunk_ids), dtype="float32")
        for start in range(0, len(self.chunk_ids), self.block):
            stop = min(start + self.block, len(self.chunk_ids))
            block = torch.from_numpy(np.ascontiguousarray(self.vectors[start:stop])).to(self.device)
            out[start:stop] = (block.float() @ needle).cpu().numpy()
            del block
        return out

    def close(self):
        self.resident = None
        self.embedder.close()
        if hasattr(self.vectors, "_mmap"):
            self.vectors._mmap.close()
