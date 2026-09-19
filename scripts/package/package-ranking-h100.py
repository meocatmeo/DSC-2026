"""Build the H100 upload bundle from an explicit list; never include credentials."""
import hashlib
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "analysis/ranking-h100/ranking-h100-ready.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    for folder in ("legalir2", "tests2"):
        files.update({p.relative_to(root).as_posix(): p for p in (root / folder).glob("*.py")})
    for name in ("scripts/gpu/gpu-ranking-h100.sh", "scripts/gpu/gpu-ranking-h100-runtime.sh", "docs/runbooks/RUNBOOK-ranking-h100.md",
                 "docs/runbooks/RUNBOOK-ranking.md", "data/public-official.json", "artifacts/splits/train.json",
                 "artifacts/splits/validation.json"):
        files[name] = root / name
    baseline = root / "analysis/v11/investigation/baselines"
    for name in ("v9-ensemble.run.json", "v9-ensemble.diagnostics.json", "v9-ensemble.metrics.json",
                 "v9-stacker-ens.json", "submission-v9.json"):
        files[f"analysis/ranking-h100/baseline/{name}"] = baseline / name
    files["analysis/ranking-h100/baseline/corpus.jsonl.gz"] = root / "analysis/v11/baseline/corpus.jsonl.gz"
    manifest = {}
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, source in sorted(files.items()):
            data = source.read_bytes()
            if source.suffix in (".py", ".sh", ".md"):
                data = data.replace(b"\r\n", b"\n")
            manifest[name] = hashlib.sha256(data).hexdigest()
            entry = zipfile.ZipInfo(name)
            entry.create_system = 3
            entry.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            entry.compress_type = zipfile.ZIP_STORED if name.endswith(".gz") else zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
        checksums = "".join(f"{value}  {name}\n" for name, value in manifest.items())
        archive.writestr("ranking-h100-files.sha256", checksums.encode())
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(files) | {"ranking-h100-files.sha256"}
        for name, expected in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
        assert all(not name.endswith((".conf", ".exe", ".safetensors", ".bin")) for name in archive.namelist())
    temporary.replace(output)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    print(f"Verified {len(files)} files, {output.stat().st_size / 1e6:.1f} MB: {output}")
    print(f"SHA256 {checksum}")


if __name__ == "__main__":
    main()
