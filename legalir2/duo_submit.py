"""Explicitly approved submission of an inconclusive comparator improvement.

Keep the trained duo.py byte-identical so existing checkpoints remain resumable.
Only the submit handler changes; its CLI still runs all frozen input guards.
The measured selection report is never edited.
"""
import math
import sys

from . import duo
from .h100 import score_row
from .store import Store, file_hash, read_json, write_json


def score_public_row(model, qid, record, mode):
    # Public JSON contains explicit null answers, not only absent answer keys.
    # Normalise the scoring copy; retain the original file and resume digest.
    return score_row(model, qid, dict(record, answer=record.get("answer") or []), mode)


def collect_public(args):
    original = duo.score_row
    try:
        duo.score_row = score_public_row
        return duo.collect(args, "public")
    finally:
        duo.score_row = original


def submit_approved(args):
    choice = read_json(args.out / "calibration.json")
    selection = read_json(args.out / "selection.json")
    duo.check_choice(args, choice)
    if selection.get("calibration") != file_hash(args.out / "calibration.json"):
        raise ValueError("Selection/calibration mismatch")
    strength = selection["strength"]
    if strength != choice["chosen"]["strength"] or strength not in duo.STRENGTHS or strength <= 0:
        raise ValueError("Approved submission must use the development-selected comparator strength")
    comparison = selection["comparison"]
    if (not math.isfinite(comparison["delta"]) or comparison["delta"] <= 0 or
            comparison["questions"] != len(read_json(args.validation))):
        raise ValueError("Approved submission requires the completed positive validation comparison")
    approval = {"mode": "user-approved-inconclusive", "strength": strength,
                "automatic_promotion": selection["promoted"], "comparison": comparison,
                "selection_sha256": file_hash(args.out / "selection.json"),
                "calibration_sha256": file_hash(args.out / "calibration.json"),
                "model_sha256": file_hash(args.out / "model/duo.json"),
                "submission_code_sha256": file_hash(__file__)}
    write_json(args.out / "submission-approval.json", approval)
    print(f"USER-APPROVED SUBMISSION: strength={strength}; validation delta={comparison['delta']:+.6f}; "
          f"95% CI={comparison['ci']}. Original selection report preserved.", flush=True)
    rows = collect_public(args)
    result = duo.diagnostics(rows, strength)
    write_json(args.out / "submission-duo.json", {q: {"answer": r["predicted"]} for q, r in result.items()})
    store = Store(args.index / "store.sqlite")
    try:
        temporary = args.out / "submission-duo.zip.tmp"
        duo.pack(args.out / "submission-duo.json", store, args.public, temporary)
        temporary.replace(args.out / "submission-duo.zip")
    finally:
        store.close()
    write_json(args.out / "submission-complete.json", {
        "approval_sha256": file_hash(args.out / "submission-approval.json"),
        "submission_sha256": file_hash(args.out / "submission-duo.zip"), "questions": len(result)})
    print(f"USER-APPROVED COMPARATOR SUBMISSION: {args.out / 'submission-duo.zip'}", flush=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "submit":
        raise SystemExit("Use gpu-duo-h100.sh submit-approved, or this module's submit command with the usual duo options.")
    original = duo.submit
    try:
        # Reuse the frozen parser and all checkpoint/package/question guards.
        duo.submit = submit_approved
        duo.main()
    finally:
        duo.submit = original


if __name__ == "__main__":
    main()
