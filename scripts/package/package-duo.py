"""Package an additive overlay; preserve the H100 source run's immutable files."""
import hashlib
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[2]
    names = ["legalir2/duo.py", "legalir2/impact.py", "legalir2/question_memory.py",
             "legalir2/duo_submit.py", "tests2/test_duo_submit.py",
             "tests2/test_duo.py", "tests2/test_duo_shell.py", "tests2/test_impact.py",
             "scripts/gpu/gpu-duo-h100.sh", "scripts/gpu/gpu-duo-runtime.sh", "docs/runbooks/RUNBOOK-duo.md"]
    output = root / "analysis/impact/duo-overlay.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    checksums = {}
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            data = (root / name).read_bytes().replace(b"\r\n", b"\n")
            checksums[name] = hashlib.sha256(data).hexdigest()
            entry = zipfile.ZipInfo(name)
            entry.create_system = 3
            entry.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
        manifest = "".join(f"{value}  {name}\n" for name, value in checksums.items())
        archive.writestr("duo-files.sha256", manifest.encode())
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(names) | {"duo-files.sha256"}
        for name, expected in checksums.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    temporary.replace(output)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    print(f"Verified additive overlay: {output} ({output.stat().st_size:,} bytes)")
    print(checksum)


if __name__ == "__main__":
    main()
