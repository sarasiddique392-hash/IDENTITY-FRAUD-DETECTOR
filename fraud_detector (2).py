"""
Financial Fraud Detector
Compatible with Python 3.14.6

Notes on the 3.14 update:
- Replaced `typing.Tuple` / `typing.Dict` with built-in generics (tuple[...], dict[...]),
  since PEP 585 generics are the standard going forward and typing.Tuple/Dict are
  deprecated aliases slated for eventual removal.
- No other language-level changes were needed: the script does not use any modules
  removed in 3.12-3.14 (e.g. distutils, imp, asynchat/asyncore).
- Make sure your environment has 3.14-compatible wheels installed:
    numpy>=2.1, pandas>=2.2, scikit-learn>=1.5, imbalanced-learn>=0.13
  Older pinned versions of these may not ship 3.14 wheels and will try to build
  from source, which can fail without a C/C++ toolchain.
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


class FinancialFraudDetector:
    def __init__(self, random_state: int = 42, use_smote: bool = True):
        self.random_state = random_state
        self.use_smote = use_smote and HAS_SMOTE
        self.optimal_threshold = 0.5

        class_weight = None if self.use_smote else "balanced"
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            class_weight=class_weight,
            n_jobs=1,
        )
        self.is_trained = False

    def preprocess_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )
        return X_train, X_test, y_train, y_test

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if self.use_smote:
            print("[*] Resampling training set with SMOTE...")
            smote = SMOTE(random_state=self.random_state)
            X_fit, y_fit = smote.fit_resample(X_train, y_train)
        else:
            X_fit, y_fit = X_train, y_train

        print("[*] Training RandomForest Classifier...")
        self.model.fit(X_fit, y_fit)

        train_probs = self.model.predict_proba(X_train)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_train, train_probs)

        f1_scores = np.divide(
            2 * (precisions * recalls),
            (precisions + recalls),
            out=np.zeros_like(precisions),
            where=(precisions + recalls) != 0,
        )

        if len(thresholds) > 0:
            best_idx = np.argmax(f1_scores[:-1])
            self.optimal_threshold = float(thresholds[best_idx])

        self.is_trained = True
        print("[+] Training completed successfully.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Call fit() before evaluating.")

        probabilities = self.model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= self.optimal_threshold).astype(int)

        return {
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "classification_report": classification_report(y_test, predictions, output_dict=True),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "threshold_used": self.optimal_threshold,
        }


def generate_synthetic_transactions(n_samples: int = 15000) -> pd.DataFrame:
    # NOTE: dtypes are pinned explicitly (np.int64 / np.float64) everywhere below.
    # Without this, plain `.astype(int)` or un-cast RNG output can land on a
    # platform-dependent width (32-bit on Windows vs 64-bit on Linux/Mac), and
    # pandas >= 3.0 raises a TypeError on `.loc` assignment instead of silently
    # upcasting the column like pandas 2.x used to. Pinning dtypes avoids that
    # entirely, on every OS.
    rng = np.random.default_rng(42)
    amounts = (rng.exponential(scale=45, size=n_samples) + 2).astype(np.float64)
    hours = rng.integers(0, 24, size=n_samples, dtype=np.int64)
    distance = rng.lognormal(mean=1.2, sigma=0.8, size=n_samples).astype(np.float64)

    df = pd.DataFrame({
        "amount": pd.array(amounts, dtype="float64"),
        "hour_of_day": pd.array(hours, dtype="int64"),
        "distance_from_home": pd.array(distance, dtype="float64"),
        "is_fraud": pd.array(np.zeros(n_samples), dtype="int64"),
    })

    fraud_size = int(n_samples * 0.012)
    fraud_indices = rng.choice(n_samples, size=fraud_size, replace=False)

    df.loc[fraud_indices, "is_fraud"] = np.int64(1)
    df.loc[fraud_indices, "amount"] = (
        df.loc[fraud_indices, "amount"].to_numpy() * rng.uniform(8, 20, size=fraud_size)
    ).astype(np.float64)
    df.loc[fraud_indices, "distance_from_home"] = (
        df.loc[fraud_indices, "distance_from_home"].to_numpy() * rng.uniform(12, 35, size=fraud_size)
    ).astype(np.float64)
    df.loc[fraud_indices, "hour_of_day"] = rng.choice(
        [1, 2, 3, 4], size=fraud_size
    ).astype(np.int64)

    return df


def main() -> None:
    raw_data = generate_synthetic_transactions()
    print(f"Loaded dataset of {len(raw_data)} transactions.")

    counts = raw_data["is_fraud"].value_counts()
    print(f"Legit count: {counts[0]} | Fraud count: {counts[1]}\n")

    detector = FinancialFraudDetector()
    X_train, X_test, y_train, y_test = detector.preprocess_data(raw_data, target_col="is_fraud")

    detector.fit(X_train, y_train)
    results = detector.evaluate(X_test, y_test)

    print("\n" + "=" * 40 + "\nMODEL EVALUATION RESULTS\n" + "=" * 40)
    print(f"ROC-AUC Score: {results['roc_auc']:.4f}")
    print(f"Optimal Threshold Used: {results['threshold_used']:.4f}\n")
    print("Confusion Matrix:")
    print(results["confusion_matrix"])


if __name__ == "__main__":
    main()
