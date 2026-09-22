# Frozen best-result handoff: private 0.9523

Release: `releases/handoff/20260922-best-09523-v1`.
Keep this version intact until a different submission actually beats 0.9523.
Do not merge deadline trial policies into this handoff.

## What was successfully reproduced

The recovery program recalculated all 2,080 private predictions from the two
saved comparator score matrices using the unchanged 0.90 fifth-slot agreement
rule. It reproduced all 306 fifth-slot replacements and the original ZIP bytes:

`37eb1aedbb06620f429c72c6c02de299d5452e628faa44aff816e255c45e2d3f`

Recovery passed on Windows and in a Linux/amd64 Docker container, with networking
disabled, a read-only root filesystem, all capabilities dropped and a non-root
user. The runtime computes answers from model scores; it does not copy an
embedded answer ZIP. The original ZIP is preserved separately as the reference.

**This is exact saved-score recovery, not fresh GPU model inference.** It
explicitly rejects changed/new question files. The organizer's detailed rules
have been requested. Do not present cached-score replay as proof of full
questions-to-predictions inference or assume the organizer accepts it.

## Load and recover

From the release folder on a Linux/amd64 Docker host:

```bash
sha256sum -c RELEASE.sha256
docker load -i dsc-best-09523-recovery-v1.docker.tar
mkdir -p recovered
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD/recovered,dst=/output" \
  dsc-best:09523-recovery-v1
sha256sum recovered/submission-private-agreement.zip
```

The image ID is
`sha256:84d031593e47671cf346f12c9d61684188c9eb9aee63261ea1c3682516fc7a27`.
Output includes the recomputed `submission.json`, byte-identical ZIP, and a
`recovery-report.json`. For a rebuild, the Dockerfile pins the official Python
base image by digest and installs NumPy 2.2.6 from a bundled hash-checked Linux
wheel. After the base image is available, building needs no network:

```bash
docker build --network none -t dsc-best:09523-recovery-v1 .
```

The original exported image remains the tested artifact even if a later Docker
build has a different image ID. Container build timestamps may change image
identity without changing the verified prediction payload.

## Contents and protection

- `source/`: frozen published training/inference source, exact agreement source,
  missing-import repairs, convergence validation wrapper, and packaging tools.
- `recovery/`: standalone recomputation runtime, score-only shards, private
  questions, corpus IDs, and hash manifest. No private labels.
- `reference/`: original best ZIP, result receipt and provenance.
- `model-vault/`: embedding, G32, partner Qwen, original duo and retrained control.
  Fifty required model/index files passed checks against recorded fingerprints.
- `index-vault/`: exact original retrieval indexes and store.
- `upstream/`: immutable original training release, corpus, original full score
  caches including evidence token traces, and explicitly labeled rebuilt evidence.
- `verification/`: model checks, Docker recovery and index limitations.
- `source-code.zip`: source/build/recovery bundle; large model/index vaults and
  exported image are separate and listed by the release checksums.
- `RELEASE.sha256` and `release-lock.json`: complete file inventory and checksums.

Packaging uses explicit source directories and a Docker context allowlist.
No rclone configuration, API credentials, repository history, personal home
directory, or deadline trial policies are included. A checksum lock detects
changes; it is not encryption or a storage-level write-once guarantee.

## Fresh GPU inference remains to verify

Both comparator checkpoints and all three retrieval model sets have been
downloaded and verified. Source and original score/evidence traces are retained.
Counting the additional retrained comparator gives 2,922,840,065 recorded model
parameters across the five components, below the previous 4-billion budget;
organizer interpretation of the budget is a separate rules question.

The local full-text SQLite rebuild has metadata matching the original corpus,
but its file SHA256 is
`3228f7cd040d8270b4bd57396295c82ca45acf17e00cac35231d5190a558bc9a`
instead of the original GPU index's
`9d3d85b1425c9fee8744c5aaa3a85cf8e2d6a8f678b3566d4cd8d46fa8f182f6`.
It is stored as `upstream/rebuilt-evidence-windows.sqlite`, not silently
substituted for the historical index. Original token traces remain available.
SQLite byte differences alone do not prove different retrieval behavior, but
metadata equality alone does not prove identical behavior either.

The archived GPU inference uses BF16, SDPA for V9 and FlashAttention2 for the
comparators, with the pinned packages in the published source. There is no
quantization. Fresh end-to-end inference has not been tested in this handoff.
If required by the organizer, the remaining work is a GPU inference image,
fresh retrieval/evidence equivalence checks, and an end-to-end output comparison.

The temporary build follows Docker's official
[binary installation](https://docs.docker.com/engine/install/binaries/) and
[isolated daemon configuration](https://docs.docker.com/engine/daemon/)
documentation. No system Docker service was installed.
