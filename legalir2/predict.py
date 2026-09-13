"""Run the pipeline over a question file, score it, and package a submission."""
import json
import time
import zipfile
from pathlib import Path

from .store import read_json, write_json


def load_questions(path, labeled=False):
    records = read_json(path)
    if not isinstance(records, dict) or not records:
        raise ValueError("Questions must be a non-empty JSON object")
    for key, value in records.items():
        if not isinstance(value, dict) or not isinstance(value.get("question"), str):
            raise ValueError(f"Invalid question record: {key}")
        if labeled:
            answers = value.get("answer")
            if not isinstance(answers, list) or not answers or any(not isinstance(x, str) for x in answers):
                raise ValueError(f"Invalid labels for question {key}")
    return records


def metrics(predictions, truth):
    """Macro Recall and Precision, matching the organiser's scorer including its
    zero penalty for an empty answer or more than five ids."""
    recalls, precisions = [], []
    for key, gold in truth.items():
        answer = predictions.get(key, {}).get("answer", [])
        if not answer or len(answer) > 5:
            recalls.append(0.0)
            precisions.append(0.0)
            continue
        overlap = len(set(gold) & set(answer))
        recalls.append(overlap / len(gold))
        precisions.append(overlap / len(answer))
    count = max(1, len(recalls))
    return {"recall": sum(recalls) / count, "precision": sum(precisions) / count,
            "questions": len(recalls)}


def run(pipeline, questions_path, output, limit=None, evaluate=False, top_k=5,
        progress=100):
    records = load_questions(questions_path, labeled=evaluate)
    items = list(records.items())[:limit] if limit else list(records.items())
    predictions, diagnostics = {}, {}
    started = time.time()
    for number, (qid, record) in enumerate(items, 1):
        detail = {}
        answer = pipeline.search(record["question"], top_k=top_k, diagnostics=detail)
        predictions[qid] = {"answer": list(answer)}
        if evaluate:
            gold = record["answer"]
            ranks = {g: (detail["candidates"].index(g) + 1 if g in detail["candidates"] else None)
                     for g in gold}
            diagnostics[qid] = {"question": record["question"], "relevant": gold,
                                "candidate_ranks": ranks, "predicted": list(answer)}
        if number % progress == 0:
            rate = number / max(1e-6, time.time() - started)
            print(f"  {number}/{len(items)} questions, {rate:.2f} q/s, "
                  f"eta {(len(items) - number) / max(rate, 1e-9) / 60:.1f} min", flush=True)
    output = Path(output)
    write_json(output, predictions)
    manifest = pipeline.manifest()
    manifest.update(questions=str(questions_path), count=len(items),
                    seconds=round(time.time() - started))
    write_json(output.with_suffix(".run.json"), manifest)
    if evaluate:
        truth = {k: v["answer"] for k, v in items}
        result = metrics(predictions, truth)
        candidate = _candidate_recall(diagnostics)
        result["candidate_recall"] = candidate
        write_json(output.with_suffix(".metrics.json"), result)
        write_json(output.with_suffix(".diagnostics.json"), diagnostics)
        print(f"Recall {result['recall']:.4f}  Precision {result['precision']:.4f}  "
              f"candidate recall {candidate}")
        return result
    print(f"Wrote {len(predictions)} predictions to {output}")
    return {"questions": len(predictions)}


def _candidate_recall(diagnostics, depths=(5, 10, 20, 50, 100, 200)):
    out = {}
    for depth in depths:
        total = 0.0
        for record in diagnostics.values():
            gold = record["relevant"]
            hit = sum(1 for g in gold if record["candidate_ranks"].get(g)
                      and record["candidate_ranks"][g] <= depth)
            total += hit / len(gold)
        out[f"@{depth}"] = round(total / max(1, len(diagnostics)), 4)
    return out


def validate(predictions_path, questions_path):
    predictions = read_json(predictions_path)
    questions = load_questions(questions_path)
    if set(predictions) != set(questions):
        missing = len(set(questions) - set(predictions))
        extra = len(set(predictions) - set(questions))
        raise ValueError(f"Question coverage mismatch: {missing} missing, {extra} unexpected")
    for key, value in predictions.items():
        answer = value.get("answer")
        if not isinstance(answer, list) or any(not isinstance(x, str) for x in answer):
            raise ValueError(f"{key}: answer must be a list of strings")
        if len(answer) > 5:
            raise ValueError(f"{key}: more than five document ids scores zero")
        if len(set(answer)) != len(answer):
            raise ValueError(f"{key}: duplicate document ids")
    return {"questions": len(predictions)}


def pack(predictions_path, store, questions_path, output):
    result = validate(predictions_path, questions_path)
    known = set(store.doc_ids)
    predictions = read_json(predictions_path)
    for key, value in predictions.items():
        unknown = [x for x in value["answer"] if x not in known]
        if unknown:
            raise ValueError(f"{key}: unknown document ids {unknown[:3]}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(predictions, ensure_ascii=False)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", payload.encode("utf-8"))
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        if names != ["submission.json"]:
            raise ValueError(f"Unexpected archive contents: {names}")
        json.loads(archive.read("submission.json").decode("utf-8"))
    print(f"Packed {result['questions']} predictions into {output}")
    return result
