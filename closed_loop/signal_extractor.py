"""Extract early training signals from experiment history for screening."""

import numpy as np


def extract_early_signals(history: list, baseline_history: list) -> dict:
    """
    Extract early-stage training signals from an experiment's history.

    Args:
        history: list of dicts with keys train_loss, train_acc, test_loss, test_acc
        baseline_history: baseline history (same format), trimmed to same length

    Returns:
        dict of signal values for screening analysis
    """
    n = len(history)
    if n == 0:
        return _empty_signals()

    # Trim baseline to same number of epochs
    bl = baseline_history[:n]

    epochs = np.arange(n)
    train_losses = np.array([h["train_loss"] for h in history])
    test_losses = np.array([h["test_loss"] for h in history])
    train_accs = np.array([h["train_acc"] for h in history])
    test_accs = np.array([h["test_acc"] for h in history])

    bl_test_accs = np.array([h["test_acc"] for h in bl]) if bl else test_accs

    # Linear fit slopes (per-epoch change)
    train_loss_slope = _linear_slope(epochs, train_losses)
    val_loss_slope = _linear_slope(epochs, test_losses)

    # Gaps
    last_gap = float(train_accs[-1] - test_accs[-1])
    first_gap = float(train_accs[0] - test_accs[0])
    gap_trend = last_gap - first_gap

    # Best epoch (1-indexed)
    best_epoch = int(np.argmax(test_accs)) + 1

    return {
        "val_acc": float(test_accs[-1]),
        "val_acc_delta": float(test_accs[-1] - bl_test_accs[-1]) if len(bl) == n else 0.0,
        "train_loss_slope": float(train_loss_slope),
        "val_loss_slope": float(val_loss_slope),
        "train_val_gap": last_gap,
        "gap_trend": gap_trend,
        "val_loss_var": float(np.var(test_losses)),
        "best_epoch": best_epoch,
    }


def _linear_slope(x, y):
    """Compute slope of linear fit."""
    if len(x) < 2:
        return 0.0
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0]


def _empty_signals():
    return {
        "val_acc": 0.0, "val_acc_delta": 0.0,
        "train_loss_slope": 0.0, "val_loss_slope": 0.0,
        "train_val_gap": 0.0, "gap_trend": 0.0,
        "val_loss_var": 0.0, "best_epoch": 0,
    }
