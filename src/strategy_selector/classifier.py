"""Strategy classifier: predict best-fit strategy per user."""

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

_MODEL_PATH = Path("data/processed/strategy_classifier.pkl")
_LE_PATH = Path("data/processed/strategy_label_encoder.pkl")

STRATEGY_CLASSES = ["collaborative", "content_based", "popularity"]


class StrategyClassifier:
    """ML classifier for per-user strategy selection."""

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        # Predictions on the held-out split from the most recent fit(), so
        # downstream reporting (e.g. "sample predictions") reads from the same
        # population the confusion matrix was computed on, never train-set rows.
        self.holdout_predictions = None

    def fit(self, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> dict:
        """Train classifier on features and strategy labels.

        Args:
            features_df: DataFrame with [userId, num_ratings, avg_rating, ...]
            labels_df: DataFrame with [userId, best_strategy, ...]

        Returns:
            Dictionary with evaluation metrics: accuracy, majority_baseline_accuracy,
            precision/recall/f1 (weighted), per_class (precision/recall/f1/support per
            strategy), roc_auc, confusion_matrix (+ confusion_matrix_labels giving the
            row/column order), classes_present, classes_missing.
        """
        merged = features_df.merge(labels_df[["userId", "best_strategy"]], on="userId")

        if merged.empty:
            return {}

        self.feature_names = [col for col in merged.columns if col not in ("userId", "best_strategy")]
        X = merged[self.feature_names].fillna(0)
        y = merged["best_strategy"]

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        classes_present = list(self.label_encoder.classes_)
        classes_missing = [c for c in STRATEGY_CLASSES if c not in classes_present]

        stratify = y_encoded if len(y_encoded) > len(classes_present) * 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=stratify
        )

        self.model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        confidences = y_pred_proba.max(axis=1)

        test_user_ids = merged.loc[X_test.index, "userId"].values
        self.holdout_predictions = pd.DataFrame(
            {
                "userId": test_user_ids,
                "true_strategy": self.label_encoder.inverse_transform(y_test),
                "predicted_strategy": self.label_encoder.inverse_transform(y_pred),
                "confidence": confidences,
            }
        )
        self.holdout_predictions["correct"] = (
            self.holdout_predictions["true_strategy"] == self.holdout_predictions["predicted_strategy"]
        )

        # Majority-class baseline: always predict the most common training label.
        # Accuracy above this is the actual signal from features, not class imbalance.
        majority_class = Counter(y_train).most_common(1)[0][0]
        majority_baseline_accuracy = float(np.mean(y_test == majority_class))

        class_labels = list(range(len(classes_present)))
        precisions, recalls, f1s, supports = precision_recall_fscore_support(
            y_test, y_pred, labels=class_labels, zero_division=0
        )
        per_class = {
            cls_name: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for cls_name, p, r, f, s in zip(classes_present, precisions, recalls, f1s, supports)
        }

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "majority_baseline_accuracy": majority_baseline_accuracy,
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "per_class": per_class,
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=class_labels).tolist(),
            "confusion_matrix_labels": classes_present,
            "classes_present": classes_present,
            "classes_missing": classes_missing,
        }

        if len(classes_present) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
        elif len(classes_present) > 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba, multi_class="ovr"))
            except Exception:
                metrics["roc_auc"] = 0.0

        return metrics

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict strategy for users given their profile features.

        Args:
            features_df: DataFrame with [userId, num_ratings, avg_rating, ...]

        Returns:
            DataFrame with [userId, predicted_strategy, confidence]
        """
        if self.model is None or self.label_encoder is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = features_df[self.feature_names].fillna(0)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X).max(axis=1)

        result = pd.DataFrame(
            {
                "userId": features_df["userId"],
                "predicted_strategy": self.label_encoder.inverse_transform(predictions),
                "confidence": probabilities,
            }
        )
        return result

    def save(self) -> None:
        """Persist model and label encoder to disk."""
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(_LE_PATH, "wb") as f:
            pickle.dump((self.label_encoder, self.feature_names), f)

    def load(self) -> bool:
        """Load persisted model from disk. Return True if successful."""
        if not _MODEL_PATH.exists() or not _LE_PATH.exists():
            return False
        try:
            with open(_MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            with open(_LE_PATH, "rb") as f:
                self.label_encoder, self.feature_names = pickle.load(f)
            return True
        except Exception:
            return False
