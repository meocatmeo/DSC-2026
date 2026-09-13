"""Fetch approved checkpoints over HTTPS and record immutable provenance.

`download` is the only command that touches the network. It resolves the branch to
a commit sha, records a SHA-256 for every file it writes, and stores both in
`provenance.json`. Every other command loads local files only, so a run can be
reproduced from the archived directories without any network access.
"""
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .store import write_json

ENDPOINT = "https://huggingface.co"

# Files a local transformers load needs. Weight shards are discovered from the index.
WANTED = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "sentencepiece.bpe.model", "spm.model", "vocab.txt", "vocab.json", "merges.txt",
          "model.safetensors", "model.safetensors.index.json", "generation_config.json",
          "sentence_bert_config.json", "modules.json", "added_tokens.json")


def _request(url):
    request = urllib.request.Request(url, headers={"User-Agent": "legalir2/1.0"})
    return urllib.request.urlopen(request, timeout=120)


def resolve(repo, revision="main"):
    with _request(f"{ENDPOINT}/api/models/{repo}/revision/{revision}") as response:
        payload = json.load(response)
    return payload["sha"], [f["rfilename"] for f in payload.get("siblings", [])]


def fetch(repo, sha, name, target):
    url = f"{ENDPOINT}/{repo}/resolve/{sha}/{name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    with _request(url) as response, partial.open("wb") as stream:
        while block := response.read(1 << 20):
            digest.update(block)
            stream.write(block)
    partial.replace(target)
    return digest.hexdigest(), target.stat().st_size


def download(repo, directory, revision="main"):
    directory = Path(directory)
    sha, names = resolve(repo, revision)
    shards = [n for n in names if n.startswith("model-") and n.endswith(".safetensors")]
    wanted = [n for n in names if n in WANTED or n in shards]
    if not any(n.endswith(".safetensors") for n in wanted):
        # Some approved checkpoints only publish PyTorch bins.
        wanted += [n for n in names if n in ("pytorch_model.bin", "pytorch_model.bin.index.json")]
        wanted += [n for n in names if n.startswith("pytorch_model-") and n.endswith(".bin")]
    if "config.json" not in wanted:
        raise ValueError(f"{repo} does not publish config.json at revision {sha}")
    files, started = {}, time.time()
    for name in sorted(set(wanted)):
        target = directory / name
        try:
            digest, size = fetch(repo, sha, name, target)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue
            raise
        files[name] = {"sha256": digest, "bytes": size}
        print(f"  {repo}: {name} {size / 1e6:.1f} MB", flush=True)
    provenance = {"repository": repo, "revision": revision, "commit": sha, "files": files,
                  "endpoint": ENDPOINT, "seconds": round(time.time() - started)}
    write_json(directory / "provenance.json", provenance)
    return provenance


def verify(directory):
    directory = Path(directory)
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    problems = []
    for name, record in provenance["files"].items():
        path = directory / name
        if not path.is_file():
            problems.append(f"missing {name}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            problems.append(f"hash mismatch {name}")
    if problems:
        raise ValueError(f"{directory}: " + "; ".join(problems))
    return provenance
