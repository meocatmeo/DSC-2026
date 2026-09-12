"""Competition metrics and strict submission packaging."""

import json
import zipfile
from pathlib import Path

from .data import read_json, unique_object


def validate(predictions, questions, document_ids):
    if not isinstance(predictions, dict):
        raise ValueError("Submission must be a JSON object")
    if set(predictions) != set(questions):
        missing = sorted(set(questions) - set(predictions))[:5]
        extra = sorted(set(predictions) - set(questions))[:5]
        raise ValueError(f"Question ID mismatch; missing={missing}, extra={extra}")
    for key, record in predictions.items():
        if not isinstance(record, dict) or set(record) != {"answer"}:
            raise ValueError(f"{key}: expected exactly the answer field")
        answer = record["answer"]
        if not isinstance(answer, list) or any(not isinstance(x, str) for x in answer):
            raise ValueError(f"{key}: answer must be an array of strings")
        if len(answer) > 5 or len(answer) != len(set(answer)):
            raise ValueError(f"{key}: more than five IDs or duplicate IDs")
        if not set(answer) <= document_ids:
            raise ValueError(f"{key}: unknown document IDs: {set(answer) - document_ids}")
    return {"questions": len(predictions), "empty_answers": sum(not x["answer"] for x in predictions.values())}


def evaluate(predictions, truth):
    if not truth or set(predictions) != set(truth):
        raise ValueError("Prediction/truth IDs must match exactly and be nonempty")
    recall, precision = 0.0, 0.0
    per_question = {}
    for key, row in truth.items():
        relevant = row["answer"] if isinstance(row, dict) else row
        predicted = predictions[key]["answer"]
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"{key}: missing ground truth; public test cannot be scored locally")
        if not isinstance(predicted, list):
            raise ValueError(f"{key}: answer must be an array")
        r = p = 0.0
        if 0 < len(predicted) <= 5:
            hits = len(set(relevant) & set(predicted))
            r, p = hits / len(relevant), hits / len(predicted)
        recall += r
        precision += p
        per_question[key] = {"recall": r, "precision": p}
    return {"recall": recall / len(truth), "precision": precision / len(truth),
            "questions": len(truth), "per_question": per_question}


def pack(source, destination, questions, document_ids):
    predictions = read_json(source)
    report = validate(predictions, questions, document_ids)
    destination = Path(destination)
    if destination.name != "submission.zip":
        raise ValueError("Output ZIP must be named submission.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(predictions, ensure_ascii=False, indent=2).encode("utf-8")
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        entry = zipfile.ZipInfo("submission.json", date_time=(2026, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(entry, payload)
    with zipfile.ZipFile(temporary) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise ValueError("ZIP verification failed")
        validate(json.loads(archive.read("submission.json").decode("utf-8"),
                            object_pairs_hook=unique_object), questions, document_ids)
    temporary.replace(destination)
    return report