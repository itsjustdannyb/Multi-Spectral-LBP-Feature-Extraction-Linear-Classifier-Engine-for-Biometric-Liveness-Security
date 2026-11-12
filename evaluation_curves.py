from __future__ import annotations

from pathlib import Path
import time

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import auc, det_curve, roc_curve

from lbp_dataset import get_lbp_and_histogram


def compute_eer(
    fpr: np.ndarray, fnr: np.ndarray, thresholds: np.ndarray
) -> tuple[float, float]:
    """Return (eer, threshold) where |FPR - FNR| is minimized."""
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float(0.5 * (fpr[idx] + fnr[idx]))
    thresh = float(thresholds[idx]) if idx < len(thresholds) else float("nan")
    return eer, thresh


def benchmark_frame_latencies(
    test_df: pd.DataFrame,
    dataset_path: str | Path,
    model,
    n_samples: int = 300,
    warmup: int = 5,
) -> dict[str, np.ndarray]:
    """Measure per-frame LBP extraction, classifier inference, and total latencies (ms)."""
    root = Path(dataset_path)
    count = min(n_samples, len(test_df))
    indices = np.linspace(0, len(test_df) - 1, count, dtype=int)
    paths = [root / test_df.iloc[i]["image"] for i in indices]

    for path in paths[:warmup]:
        res = get_lbp_and_histogram(path)
        if res is not None:
            model.predict_proba(res[1][np.newaxis, :])

    extract_ms: list[float] = []
    infer_ms: list[float] = []
    total_ms: list[float] = []

    for path in paths:
        t0 = time.perf_counter()
        res = get_lbp_and_histogram(path)
        t1 = time.perf_counter()
        if res is None:
            continue
        _ = model.predict_proba(res[1][np.newaxis, :])
        t2 = time.perf_counter()

        extract_ms.append((t1 - t0) * 1000.0)
        infer_ms.append((t2 - t1) * 1000.0)
        total_ms.append((t2 - t0) * 1000.0)

    return {
        "Total frame (LBP + LR)": np.array(total_ms, dtype=float),
        "LBP extraction": np.array(extract_ms, dtype=float),
        "Classifier inference": np.array(infer_ms, dtype=float),
    }


def _plot_roc_panel(
    ax: plt.Axes, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, eer: float
) -> None:
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random (AUC = 0.50)")
    ax.scatter(
        [eer], [1.0 - eer], color="crimson", zorder=5, label=f"EER = {eer * 100:.2f}%"
    )
    ax.set_title("ROC Curve (Liveness Detection)")
    ax.set_xlabel("False Acceptance Rate (FPR / APCER)")
    ax.set_ylabel("Genuine Acceptance Rate (TPR)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")


def _plot_det_panel(
    ax: plt.Axes, fpr: np.ndarray, fnr: np.ndarray, eer: float
) -> None:
    eps = 1e-4
    x_det = norm.ppf(np.clip(fpr, eps, 1.0 - eps))
    y_det = norm.ppf(np.clip(fnr, eps, 1.0 - eps))
    ax.plot(x_det, y_det, lw=2, label=f"DET (EER = {eer * 100:.2f}%)")

    tick_pcts = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 40, 60, 80])
    tick_locs = norm.ppf(tick_pcts / 100.0)
    tick_labels = [f"{p:g}%" for p in tick_pcts]

    eer_q = norm.ppf(np.clip(eer, eps, 1.0 - eps))
    ax.plot(
        [tick_locs[0], tick_locs[-1]],
        [tick_locs[0], tick_locs[-1]],
        "k--",
        lw=1,
        alpha=0.5,
        label="FAR = FRR",
    )
    ax.scatter([eer_q], [eer_q], color="crimson", zorder=5)

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.set_yticks(tick_locs)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlim(tick_locs[0], tick_locs[-1])
    ax.set_ylim(tick_locs[0], tick_locs[-1])
    ax.set_title("DET Curve (Normal-Deviate Scale)")
    ax.set_xlabel("False Acceptance Rate (FAR / APCER)")
    ax.set_ylabel("False Rejection Rate (FRR / BPCER)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")


def _plot_score_cdf_panel(
    ax: plt.Axes, y_true: np.ndarray, y_scores: np.ndarray, eer_thresh: float
) -> None:
    spoof_scores = np.sort(y_scores[y_true == 0])
    real_scores = np.sort(y_scores[y_true == 1])

    if len(spoof_scores):
        cdf_spoof = np.arange(1, len(spoof_scores) + 1) / len(spoof_scores)
        ax.step(
            spoof_scores,
            cdf_spoof,
            where="post",
            lw=2,
            color="crimson",
            label="Spoof CDF (1 - FAR)",
        )
    if len(real_scores):
        cdf_real = np.arange(1, len(real_scores) + 1) / len(real_scores)
        ax.step(
            real_scores,
            cdf_real,
            where="post",
            lw=2,
            color="steelblue",
            label="Real CDF (FRR)",
        )
    if np.isfinite(eer_thresh):
        ax.axvline(
            eer_thresh,
            color="k",
            linestyle="--",
            lw=1.2,
            label=f"EER threshold = {eer_thresh:.3f}",
        )
    ax.set_title("Score CDF (Spoof vs. Real)")
    ax.set_xlabel("Predicted Liveness Score P(real | x)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)


def _plot_latency_cdf_panel(
    ax: plt.Axes,
    latencies_ms: np.ndarray | dict[str, np.ndarray],
    log_scale: bool,
) -> dict[str, float]:
    series = (
        latencies_ms
        if isinstance(latencies_ms, dict)
        else {"Total frame latency": latencies_ms}
    )
    stats: dict[str, float] = {}

    for name, vals in series.items():
        arr = np.sort(np.asarray(vals, dtype=float))
        if len(arr) == 0:
            continue
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        (line,) = ax.step(
            arr,
            cdf,
            where="post",
            lw=2,
            label=f"{name} (p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms)",
        )
        ax.axvline(p95, color=line.get_color(), linestyle=":", lw=1.2, alpha=0.75)
        stats[f"{name}_p50_ms"] = float(p50)
        stats[f"{name}_p95_ms"] = float(p95)
        stats[f"{name}_p99_ms"] = float(p99)

    if log_scale:
        ax.set_xscale("log")
    ax.set_title("Frame-Latency CDF")
    ax.set_xlabel("Latency per Frame (ms)")
    ax.set_ylabel("Cumulative Probability P(latency <= t)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, which="both" if log_scale else "major", alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    return stats


def plot_liveness_and_latency_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    latencies_ms: np.ndarray | dict[str, np.ndarray],
    save_path: str | Path | None = None,
    show: bool = True,
    log_latency_axis: bool = True,
) -> dict[str, float]:
    """Plot ROC, DET, Spoof/Real Score CDF, and Frame-Latency CDF in a 2x2 figure."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    fpr_roc, tpr_roc, _ = roc_curve(
        y_true, y_scores, pos_label=1, drop_intermediate=False
    )
    roc_auc = float(auc(fpr_roc, tpr_roc))

    fpr_det, fnr_det, det_thresh = det_curve(y_true, y_scores, pos_label=1)
    eer, eer_thresh = compute_eer(fpr_det, fnr_det, det_thresh)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    _plot_roc_panel(axes[0, 0], fpr_roc, tpr_roc, roc_auc, eer)
    _plot_det_panel(axes[0, 1], fpr_det, fnr_det, eer)
    _plot_score_cdf_panel(axes[1, 0], y_true, y_scores, eer_thresh)
    latency_stats = _plot_latency_cdf_panel(
        axes[1, 1], latencies_ms, log_scale=log_latency_axis
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    summary = {"roc_auc": roc_auc, "eer": eer, "eer_threshold": eer_thresh}
    summary.update(latency_stats)
    return summary
