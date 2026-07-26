from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features/sentence_word_features.npz")
    parser.add_argument("--model-out", default="models/sentence_ending_classifier.joblib")
    parser.add_argument("--report-out", default="models/sentence_training_report.json")
    parser.add_argument(
        "--classifier",
        choices=["logreg", "linear_svc", "sgd_log", "rbf_svc", "mlp"],
        default="logreg",
    )
    parser.add_argument(
        "--label-mode",
        choices=["exact", "base_vowel"],
        default="exact",
        help="base_vowel collapses tanween labels into a/u/i for coarse i'rab evaluation.",
    )
    args = parser.parse_args()

    features_path = ROOT / args.features
    model_out = ROOT / args.model_out
    report_out = ROOT / args.report_out

    if not features_path.exists():
        raise SystemExit(f"features not found: {features_path}")

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    data = np.load(features_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    if args.label_mode == "base_vowel":
        y = np.array([to_base_vowel(str(label)) for label in y])
    groups = data["groups"] if "groups" in data.files else np.arange(len(y))

    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        raise SystemExit("need at least two labels")
    if len(y) < 40:
        raise SystemExit("need at least 40 aligned word segments for a useful sentence model")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    clf = make_pipeline(StandardScaler(), build_classifier(args.classifier))
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])

    report = {
        "n_samples": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "labels": {str(label): int(count) for label, count in zip(labels, counts)},
        "split": "GroupShuffleSplit by utterance_id",
        "classifier": args.classifier,
        "label_mode": args.label_mode,
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "classification_report": classification_report(
            y[test_idx], pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": sorted(set(y.tolist())),
        "confusion_matrix": confusion_matrix(
            y[test_idx], pred, labels=sorted(set(y.tolist()))
        ).tolist(),
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"accuracy: {report['accuracy']:.1%}")
    print(f"samples: {report['n_samples']} train={report['n_train']} test={report['n_test']}")
    print(f"saved: {model_out}")
    print(f"saved: {report_out}")


def build_classifier(name: str):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC

    if name == "linear_svc":
        return CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced", max_iter=5000, dual="auto"),
            cv=3,
        )
    if name == "sgd_log":
        return SGDClassifier(
            loss="log_loss",
            alpha=0.0001,
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )
    if name == "rbf_svc":
        return make_pipeline(
            PCA(n_components=192, random_state=42),
            SVC(C=4.0, gamma="scale", class_weight="balanced", probability=True),
        )
    if name == "mlp":
        return make_pipeline(
            PCA(n_components=256, random_state=42),
            MLPClassifier(
                hidden_layer_sizes=(192, 96),
                alpha=0.0008,
                batch_size=128,
                early_stopping=True,
                max_iter=500,
                random_state=42,
            ),
        )
    return LogisticRegression(max_iter=3000, class_weight="balanced")


def to_base_vowel(label: str) -> str:
    if label in {"damma", "tanwin_damm"}:
        return "damma"
    if label in {"fatha", "tanwin_fath"}:
        return "fatha"
    if label in {"kasra", "tanwin_kasr"}:
        return "kasra"
    return "none"


if __name__ == "__main__":
    main()
