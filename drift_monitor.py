"""
Feature Drift Monitor
Detects distribution shift in user and ad features using KL-divergence.
Raises alerts when drift exceeds configurable thresholds.
"""

import numpy as np
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class DriftAlert:
    feature_name: str
    kl_divergence: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    severity: str = "warning"   # 'warning' | 'critical'

    def __str__(self):
        return (
            f"[{self.severity.upper()}] {self.feature_name}: "
            f"KL={self.kl_divergence:.4f} (threshold={self.threshold:.4f})"
        )


@dataclass
class DriftReport:
    timestamp: float
    num_features_checked: int
    num_drifted: int
    alerts: List[DriftAlert]
    kl_scores: Dict[str, float]

    def summary(self) -> str:
        lines = [
            f"Drift Report @ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            f"  Features checked : {self.num_features_checked}",
            f"  Features drifted : {self.num_drifted}",
        ]
        if self.alerts:
            lines.append("  Alerts:")
            for a in self.alerts:
                lines.append(f"    {a}")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
# Core KL utilities
# ------------------------------------------------------------------ #

def _safe_kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute KL divergence D_KL(p || q) safely.
    Both arrays are normalised to sum to 1 before computation.
    eps prevents log(0).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))


def numerical_kl(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Estimate KL divergence for a continuous feature by binning.
    Uses the reference distribution's quantiles as bin edges so
    the reference histogram is roughly uniform — maximising sensitivity
    to shape changes in the current distribution.
    """
    if len(reference) < 10 or len(current) < 10:
        return 0.0

    # Bin edges from reference quantiles
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(reference, quantiles)
    edges[0]  -= 1e-9
    edges[-1] += 1e-9

    p, _ = np.histogram(reference, bins=edges, density=False)
    q, _ = np.histogram(current,   bins=edges, density=False)

    return _safe_kl_divergence(p.astype(float), q.astype(float))


def categorical_kl(
    reference: List,
    current: List,
    vocab: Optional[List] = None,
) -> float:
    """
    Compute KL divergence for a categorical feature using frequency counts.

    Args:
        reference: List of reference values
        current:   List of current values
        vocab:     Explicit vocabulary; unknown values map to an OOV bucket
    """
    all_vals = set(reference) | set(current)
    if vocab:
        all_vals = set(vocab) | {"__OOV__"}

    def freq(vals):
        counts = defaultdict(int)
        for v in vals:
            key = v if (vocab is None or v in vocab) else "__OOV__"
            counts[key] += 1
        return np.array([counts.get(k, 0) for k in sorted(all_vals)], dtype=float)

    p = freq(reference)
    q = freq(current)
    return _safe_kl_divergence(p, q)


# ------------------------------------------------------------------ #
# Drift monitor
# ------------------------------------------------------------------ #

class FeatureDriftMonitor:
    """
    Compares a reference (training) feature distribution against
    a live (serving) distribution and raises alerts on significant drift.

    Typical usage:
        monitor = FeatureDriftMonitor(warn_threshold=0.1, crit_threshold=0.3)
        monitor.fit_reference(training_samples)          # once, after training
        # ... later in production ...
        report = monitor.check(live_samples)
        if report.num_drifted > 0:
            print(report.summary())
    """

    def __init__(
        self,
        warn_threshold: float = 0.1,
        crit_threshold: float = 0.3,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        n_bins: int = 20,
        alert_callback: Optional[Callable[[DriftAlert], None]] = None,
    ):
        """
        Args:
            warn_threshold: KL divergence above this triggers a warning
            crit_threshold: KL divergence above this triggers a critical alert
            numerical_cols: Names of numerical features
            categorical_cols: Names of categorical features
            n_bins: Histogram bins for numerical KL estimation
            alert_callback: Optional function called on each alert
        """
        self.warn_threshold = warn_threshold
        self.crit_threshold = crit_threshold
        self.numerical_cols  = numerical_cols  or [f"I{i}" for i in range(1, 14)]
        self.categorical_cols = categorical_cols or [f"C{i}" for i in range(1, 7)]
        self.n_bins = n_bins
        self.alert_callback = alert_callback

        # Reference distributions (set by fit_reference)
        self._ref_numerical:    Dict[str, np.ndarray] = {}
        self._ref_categorical:  Dict[str, List]       = {}
        self._is_fitted = False

        # History of reports
        self._history: List[DriftReport] = []

    # ---------------------------------------------------------------- #
    # Fitting
    # ---------------------------------------------------------------- #

    def fit_reference(self, samples: List[Dict]):
        """
        Record reference distributions from training / validation samples.

        Args:
            samples: List of feature dicts, each with 'numerical' and 'categorical' keys
        """
        num_data: Dict[str, List] = defaultdict(list)
        cat_data: Dict[str, List] = defaultdict(list)

        for s in samples:
            for col in self.numerical_cols:
                val = s.get("numerical", {}).get(col)
                if val is not None:
                    num_data[col].append(float(val))
            for col in self.categorical_cols:
                val = s.get("categorical", {}).get(col)
                if val is not None:
                    cat_data[col].append(str(val))

        self._ref_numerical   = {k: np.array(v) for k, v in num_data.items()}
        self._ref_categorical = dict(cat_data)
        self._is_fitted = True
        logger.info(
            f"Reference fitted on {len(samples)} samples — "
            f"{len(self._ref_numerical)} numerical, "
            f"{len(self._ref_categorical)} categorical features."
        )

    def save_reference(self, path: str):
        """Persist reference distributions to disk."""
        data = {
            "numerical":   {k: v.tolist() for k, v in self._ref_numerical.items()},
            "categorical": self._ref_categorical,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Reference saved → {path}")

    def load_reference(self, path: str):
        """Load persisted reference distributions."""
        with open(path) as f:
            data = json.load(f)
        self._ref_numerical   = {k: np.array(v) for k, v in data["numerical"].items()}
        self._ref_categorical = data["categorical"]
        self._is_fitted = True
        print(f"Reference loaded from {path}")

    # ---------------------------------------------------------------- #
    # Checking
    # ---------------------------------------------------------------- #

    def check(self, current_samples: List[Dict]) -> DriftReport:
        """
        Compute KL divergence for all features and return a DriftReport.

        Args:
            current_samples: Live samples with same schema as reference

        Returns:
            DriftReport
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_reference() before check().")

        # Aggregate current samples
        num_current: Dict[str, List] = defaultdict(list)
        cat_current: Dict[str, List] = defaultdict(list)

        for s in current_samples:
            for col in self.numerical_cols:
                val = s.get("numerical", {}).get(col)
                if val is not None:
                    num_current[col].append(float(val))
            for col in self.categorical_cols:
                val = s.get("categorical", {}).get(col)
                if val is not None:
                    cat_current[col].append(str(val))

        kl_scores: Dict[str, float] = {}
        alerts: List[DriftAlert] = []

        # Numerical features
        for col, ref_vals in self._ref_numerical.items():
            cur_vals = np.array(num_current.get(col, []))
            if len(cur_vals) == 0:
                continue
            kl = numerical_kl(ref_vals, cur_vals, self.n_bins)
            kl_scores[col] = kl
            alert = self._maybe_alert(col, kl)
            if alert:
                alerts.append(alert)

        # Categorical features
        for col, ref_vals in self._ref_categorical.items():
            cur_vals = cat_current.get(col, [])
            if not cur_vals:
                continue
            kl = categorical_kl(ref_vals, cur_vals)
            kl_scores[col] = kl
            alert = self._maybe_alert(col, kl)
            if alert:
                alerts.append(alert)

        report = DriftReport(
            timestamp=time.time(),
            num_features_checked=len(kl_scores),
            num_drifted=len(alerts),
            alerts=alerts,
            kl_scores=kl_scores,
        )
        self._history.append(report)
        return report

    def _maybe_alert(self, feature: str, kl: float) -> Optional[DriftAlert]:
        if kl >= self.crit_threshold:
            alert = DriftAlert(
                feature_name=feature,
                kl_divergence=kl,
                threshold=self.crit_threshold,
                severity="critical",
            )
        elif kl >= self.warn_threshold:
            alert = DriftAlert(
                feature_name=feature,
                kl_divergence=kl,
                threshold=self.warn_threshold,
                severity="warning",
            )
        else:
            return None

        if self.alert_callback:
            self.alert_callback(alert)
        return alert

    def get_history(self) -> List[DriftReport]:
        return list(self._history)

    def latest_kl_scores(self) -> Optional[Dict[str, float]]:
        """Return KL scores from the most recent check."""
        if not self._history:
            return None
        return self._history[-1].kl_scores


# ------------------------------------------------------------------ #
# Smoke test
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=== Feature Drift Monitor — smoke test ===\n")

    rng = np.random.default_rng(42)

    def _make_samples(n, shift=0.0, cat_shift=False):
        samples = []
        for _ in range(n):
            num = {f"I{i}": float(rng.normal(i + shift, 1.0)) for i in range(1, 14)}
            cat = {f"C{i}": f"cat_{rng.integers(0, 10 + (5 if cat_shift else 0))}"
                   for i in range(1, 7)}
            samples.append({"numerical": num, "categorical": cat})
        return samples

    reference = _make_samples(5000)
    no_drift   = _make_samples(500)
    drifted    = _make_samples(500, shift=3.0, cat_shift=True)

    monitor = FeatureDriftMonitor(warn_threshold=0.1, crit_threshold=0.3)
    monitor.fit_reference(reference)

    print("--- No-drift check ---")
    r1 = monitor.check(no_drift)
    print(r1.summary())

    print("\n--- Drifted check ---")
    r2 = monitor.check(drifted)
    print(r2.summary())

    print(f"\nMax KL (no drift):  {max(r1.kl_scores.values()):.4f}")
    print(f"Max KL (drifted):   {max(r2.kl_scores.values()):.4f}")
    print("\n✓ Drift monitor smoke test passed!")
