# Auto-stop for the passage-order evaluation

The helper `scripts/gpu/vast_stop_passage.py` adapts the tested retraining watcher
for `passage-order-4090-v2`. It is separate from the running GPU code and never
modifies its files. Run it in a second terminal on the rental; the default root is
`/workspace/DSC-passage-order-4090-v2`.

The current user-authorized target is instance **52083026**. Pass `--instance
52083026`; any conflicting `CONTAINER_ID` or `VAST_CONTAINERLABEL` stops arming.
A read-only Vast API request also checks the requested instance identity before
launching the watcher. Credentials are obtained from `CONTAINER_API_KEY` or a
hidden terminal prompt. They are never supplied through CLI arguments or chat.
The local watcher settings/logs are private files under `~/.config/dsc-vast-stop`.

The background watcher waits for the target run's exit marker and obtains its
workspace lock. The marker must indicate successful exit and final backup. It
checks the pinned plan identity, all 429 prepared question checksums, both replay
receipts, all 858 scored question files, and their aggregate evaluation checksum.
It then runs `rclone check --one-way` over all included local artifacts against
`gdrive:DSC-2026-results/runs/passage-order/passage-order-4090-v2/artifacts`.
Only after these checks pass does it request `state=stopped` for the specified
instance. There is no destroy operation.

A failed run, absent report, changed/incomplete scores, backup mismatch, or API
failure keeps the instance running. Setup failure before the target run starts
also requires manual attention. The helper does not stop by elapsed time, and
does not stop other running workspaces. Keep this rental dedicated to this job.

Wait for the terminal to say `AUTO-STOP ARMED for instance 52083026`. The agent
cannot claim arming succeeded until the helper actually runs on the GPU instance.
The watcher log location is printed by the helper. It continues after closing
the second terminal and refuses duplicate live watchers for the same run.

Stop preserves disk, so storage charges continue. See the official
[Vast instance management guide](https://github.com/vast-ai/docs/blob/main/guides/instances/manage-instances.mdx).
The helper reuses the existing authenticated GET/PUT instance API implementation.

Tests: `tests2/test_vast_stop_passage.py` and the original retraining watcher tests
pass 12 cases with all Vast API calls mocked. No real instance was stopped or
armed during local testing. The GPU evaluation release remains unchanged.
