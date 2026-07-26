from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the acoustic ending classifier when word segmentation is "
            "known from forced alignment/TextGrid timings. This measures the "
            "classifier separately from the live ASR/segmentation problem."
        )
    )
    parser.add_argument(
        "--features",
        default="data/features/asc_wordtier_balanced_700_tail650_features.npz",
        help="Feature matrix extracted from TextGrid word spans.",
    )
    parser.add_argument(
        "--model",
        default="models/sentence_ending_classifier_wordtier_tail650_rbf.joblib",
        help="Acoustic ending classifier trained on aligned word-ending features.",
    )
    parser.add_argument(
        "--meta",
        default="data/features/asc_wordtier_balanced_700_tail650_features_meta.json",
        help="Metadata produced beside the feature matrix.",
    )
    parser.add_argument(
        "--report-out",
        default="models/forced_alignment_oracle_report.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    features_path = ROOT / args.features
    model_path = ROOT / args.model
    meta_path = ROOT / args.meta
    report_path = ROOT / args.report_out

    for path in (features_path, model_path, meta_path):
        if not path.exists():
            raise SystemExit(f"missing required file: {path}")

    data = np.load(features_path, allow_pickle=True)
    x = data["X"]
    y = np.array([str(item) for item in data["y"]])
    groups = np.array([str(item) for item in data["groups"]]) if "groups" in data.files else np.arange(len(y))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if args.limit and args.limit > 0:
        x = x[: args.limit]
        y = y[: args.limit]
        groups = groups[: args.limit]
        meta = meta[: args.limit]

    clf = joblib.load(model_path)
    pred = np.array([str(item) for item in clf.predict(x)])

    labels = sorted(set(y.tolist()) | set(pred.tolist()))
    confusion = {gold: {label: 0 for label in labels} for gold in labels}
    per_label = {
        label: {"support": 0, "correct": 0, "precision_denominator": 0}
        for label in labels
    }
    mistakes = []

    for i, (gold, got) in enumerate(zip(y, pred)):
        confusion[gold][got] += 1
        per_label[gold]["support"] += 1
        per_label[got]["precision_denominator"] += 1
        if gold == got:
            per_label[gold]["correct"] += 1
        elif len(mistakes) < 50:
            row = meta[i] if i < len(meta) else {}
            mistakes.append(
                {
                    "utterance_id": row.get("utterance_id"),
                    "word_index": row.get("word_index"),
                    "word": row.get("word"),
                    "expected": gold,
                    "detected": got,
                    "start_ms": row.get("start_ms"),
                    "end_ms": row.get("end_ms"),
                }
            )

    correct = int(np.sum(pred == y))
    total = int(len(y))
    label_report = {}
    for label, stats in per_label.items():
        support = stats["support"]
        precision_den = stats["precision_denominator"]
        tp = stats["correct"]
        precision = tp / precision_den if precision_den else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        label_report[label] = {
            "support": int(support),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    report = {
        "task": "forced_alignment_oracle_acoustic_ending_detection",
        "meaning": (
            "This uses known TextGrid word timings as the forced-alignment step. "
            "It tests whether the acoustic classifier can identify the spoken "
            "final ending when it receives the correct word-ending audio span."
        ),
        "features": str(features_path),
        "model": str(model_path),
        "samples": total,
        "utterances": int(len(set(groups.tolist()))),
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "labels": label_report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": [[confusion[gold][got] for got in labels] for gold in labels],
        "sample_mistakes": mistakes,
        "next_live_step": (
            "Replace oracle TextGrid timings with an automatic aligner for user "
            "recordings: MFA Arabic acoustic model, or wav2vec2/CTC reference "
            "forced alignment. ASR remains only a rough transcript helper."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
