import warnings

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Comprehensive metrics for wet/dry binary classification using sklearn, scipy, and statistical tests


def _compute_tpr(preds, ys):
    """Calculate true positive rate (TPR) for binary predictions."""
    if isinstance(preds, list):
        preds = np.concatenate(preds)
    if isinstance(ys, list):
        ys = np.concatenate(ys)
    return (
        np.sum((preds == 1) & (ys == 1)) / np.sum(ys == 1) if np.sum(ys == 1) > 0 else 0
    )


def _compute_tnr(preds, ys):
    """Calculate true negative rate (TNR) for binary predictions."""
    if isinstance(preds, list):
        preds = np.concatenate(preds)
    if isinstance(ys, list):
        ys = np.concatenate(ys)
    return (
        np.sum((preds == 0) & (ys == 0)) / np.sum(ys == 0) if np.sum(ys == 0) > 0 else 0
    )


# ============================
# INDIVIDUAL METRIC FUNCTIONS
# ============================


def compute_accuracy(preds, ys, probabilities=None):
    """Compute accuracy score."""
    preds, ys, _ = prepare_arrays(preds, ys)
    return accuracy_score(ys, preds)


def compute_precision(preds, ys, probabilities=None):
    """Compute precision score."""
    preds, ys, _ = prepare_arrays(preds, ys)
    return precision_score(ys, preds, zero_division=0)


def compute_recall(preds, ys, probabilities=None):
    """Compute recall score (same as TPR)."""
    preds, ys, _ = prepare_arrays(preds, ys)
    return recall_score(ys, preds, zero_division=0)


def compute_f1_score(preds, ys, probabilities=None):
    """Compute F1 score."""
    preds, ys, _ = prepare_arrays(preds, ys)
    return f1_score(ys, preds, zero_division=0)


def compute_specificity(preds, ys, probabilities=None):
    """Compute specificity (TNR)."""
    return _compute_tnr(preds, ys)


def compute_balanced_accuracy(preds, ys, probabilities=None):
    """Compute balanced accuracy."""
    preds, ys, _ = prepare_arrays(preds, ys)
    return balanced_accuracy_score(ys, preds)


def compute_jaccard_index(preds, ys, probabilities=None):
    """Compute Jaccard index."""
    preds, ys, _ = prepare_arrays(preds, ys)
    return jaccard_score(ys, preds, zero_division=0)


def compute_matthews_corrcoef(preds, ys, probabilities=None):
    """Compute Matthews correlation coefficient."""
    preds, ys, _ = prepare_arrays(preds, ys)
    return matthews_corrcoef(ys, preds)


def compute_cohen_kappa(preds, ys, probabilities=None):
    """Compute Cohen's kappa."""
    preds, ys, _ = prepare_arrays(preds, ys)
    return cohen_kappa_score(ys, preds)


def compute_confusion_matrix_metrics(preds, ys, probabilities=None):
    """Compute all confusion matrix based metrics."""
    preds, ys, _ = prepare_arrays(preds, ys)
    tn, fp, fn, tp = confusion_matrix(ys, preds, labels=[0, 1]).ravel()

    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "positive_predictive_value": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "negative_predictive_value": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
        "false_discovery_rate": fp / (fp + tp) if (fp + tp) > 0 else 0,
        "false_omission_rate": fn / (fn + tn) if (fn + tn) > 0 else 0,
    }


# Cache for confusion matrix computation to avoid recomputing
_confusion_cache = {}


def _get_confusion_matrix_components(preds, ys):
    """Get confusion matrix components with caching."""
    preds, ys, _ = prepare_arrays(preds, ys)

    # Create a simple cache key (not perfect but works for most cases)
    cache_key = (tuple(preds.flatten()), tuple(ys.flatten()))

    if cache_key not in _confusion_cache:
        tn, fp, fn, tp = confusion_matrix(ys, preds, labels=[0, 1]).ravel()
        _confusion_cache[cache_key] = (int(tn), int(fp), int(fn), int(tp))

    return _confusion_cache[cache_key]


def compute_true_positives(preds, ys, probabilities=None):
    """Compute true positives count."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return tp


def compute_false_positives(preds, ys, probabilities=None):
    """Compute false positives count."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fp


def compute_true_negatives(preds, ys, probabilities=None):
    """Compute true negatives count."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return tn


def compute_false_negatives(preds, ys, probabilities=None):
    """Compute false negatives count."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fn


def compute_positive_predictive_value(preds, ys, probabilities=None):
    """Compute positive predictive value (precision)."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def compute_negative_predictive_value(preds, ys, probabilities=None):
    """Compute negative predictive value."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return tn / (tn + fn) if (tn + fn) > 0 else 0


def compute_false_positive_rate(preds, ys, probabilities=None):
    """Compute false positive rate."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def compute_false_negative_rate(preds, ys, probabilities=None):
    """Compute false negative rate."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fn / (fn + tp) if (fn + tp) > 0 else 0


def compute_false_discovery_rate(preds, ys, probabilities=None):
    """Compute false discovery rate."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fp / (fp + tp) if (fp + tp) > 0 else 0


def compute_false_omission_rate(preds, ys, probabilities=None):
    """Compute false omission rate."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fn / (fn + tn) if (fn + tn) > 0 else 0


def compute_chi2_test(preds, ys, probabilities=None):
    """Compute chi-square test statistics."""
    preds, ys, _ = prepare_arrays(preds, ys)
    try:
        contingency_table = confusion_matrix(ys, preds, labels=[0, 1])
        chi2_stat, chi2_p_value, _, _ = chi2_contingency(contingency_table)
        return {"chi2_statistic": chi2_stat, "chi2_p_value": chi2_p_value}
    except (ValueError, ZeroDivisionError):
        return {"chi2_statistic": np.nan, "chi2_p_value": np.nan}


def compute_chi2_statistic(preds, ys, probabilities=None):
    """Compute chi-square statistic."""
    return compute_chi2_test(preds, ys)["chi2_statistic"]


def compute_chi2_p_value(preds, ys, probabilities=None):
    """Compute chi-square p-value."""
    return compute_chi2_test(preds, ys)["chi2_p_value"]


def compute_fisher_exact_p_value(preds, ys, probabilities=None):
    """Compute Fisher's exact test p-value."""
    preds, ys, _ = prepare_arrays(preds, ys)
    try:
        contingency_table = confusion_matrix(ys, preds, labels=[0, 1])
        if np.sum(contingency_table) < 1000:
            _, fisher_p_value = fisher_exact(contingency_table)
            return fisher_p_value
        else:
            return np.nan
    except (ValueError, ZeroDivisionError):
        return np.nan


def compute_roc_auc(preds, ys, probabilities=None):
    """Compute ROC AUC score."""
    if probabilities is None:
        raise ValueError("Probabilities required for ROC AUC")
    _, ys, probs = prepare_arrays(preds, ys, probabilities)
    try:
        return roc_auc_score(ys, probs)
    except (ValueError, TypeError):
        warnings.warn("Error computing ROC AUC")
        return np.nan


def compute_average_precision(preds, ys, probabilities=None):
    """Compute average precision (PR AUC)."""
    if probabilities is None:
        raise ValueError("Probabilities required for average precision")
    _, ys, probs = prepare_arrays(preds, ys, probabilities)
    try:
        return average_precision_score(ys, probs)
    except (ValueError, TypeError):
        warnings.warn("Error computing average precision")
        return np.nan


def compute_brier_score(preds, ys, probabilities=None):
    """Compute Brier score."""
    if probabilities is None:
        raise ValueError("Probabilities required for Brier score")
    _, ys, probs = prepare_arrays(preds, ys, probabilities)
    try:
        return brier_score_loss(ys, probs)
    except (ValueError, TypeError):
        warnings.warn("Error computing Brier score")
        return np.nan


def compute_log_loss(preds, ys, probabilities=None):
    """Compute log loss."""
    if probabilities is None:
        raise ValueError("Probabilities required for log loss")
    _, ys, probs = prepare_arrays(preds, ys, probabilities)
    try:
        probs_clipped = np.clip(probs, 1e-15, 1 - 1e-15)
        return log_loss(ys, probs_clipped)
    except (ValueError, TypeError):
        warnings.warn("Error computing log loss")
        return np.nan


def compute_roc_curve(preds, ys, probabilities=None):
    """Compute ROC curve data."""
    if probabilities is None:
        raise ValueError("Probabilities required for ROC curve")
    _, ys, probs = prepare_arrays(preds, ys, probabilities)
    try:
        fpr, tpr_vals, roc_thresholds = roc_curve(ys, probs)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr_vals.tolist(),
            "thresholds": roc_thresholds.tolist(),
        }
    except (ValueError, TypeError):
        return {}


def compute_pr_curve(preds, ys, probabilities=None):
    """Compute Precision-Recall curve data."""
    if probabilities is None:
        raise ValueError("Probabilities required for PR curve")
    _, ys, probs = prepare_arrays(preds, ys, probabilities)
    try:
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(ys, probs)
        return {
            "precision": precision_vals.tolist(),
            "recall": recall_vals.tolist(),
            "thresholds": pr_thresholds.tolist(),
        }
    except (ValueError, TypeError):
        return {}


def compute_critical_success_index(preds, ys, probabilities=None):
    """Compute Critical Success Index (Threat Score)."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0


def compute_hit_rate(preds, ys, probabilities=None):
    """Compute Hit Rate (Probability of Detection)."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def compute_false_alarm_rate(preds, ys, probabilities=None):
    """Compute False Alarm Rate."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def compute_false_alarm_ratio(preds, ys, probabilities=None):
    """Compute False Alarm Ratio."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fp / (fp + tp) if (fp + tp) > 0 else 0


def compute_bias_score(preds, ys, probabilities=None):
    """Compute Bias Score (frequency bias)."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0


def compute_equitable_threat_score(preds, ys, probabilities=None):
    """Compute Equitable Threat Score."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return _compute_equitable_threat_score(tp, fp, fn, tn)


def compute_heidke_skill_score(preds, ys, probabilities=None):
    """Compute Heidke Skill Score."""
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return _compute_heidke_skill_score(tp, fp, fn, tn)


def compute_tpr(preds, ys, probabilities=None):
    """Compute True Positive Rate (legacy)."""
    return _compute_tpr(preds, ys)


def compute_tnr(preds, ys, probabilities=None):
    """Compute True Negative Rate (legacy)."""
    return _compute_tnr(preds, ys)


# ============================
# METRIC REGISTRY
# ============================

METRIC_FUNCTIONS = {
    # Basic metrics
    "accuracy": compute_accuracy,
    "precision": compute_precision,
    "recall": compute_recall,
    "f1_score": compute_f1_score,
    "specificity": compute_specificity,
    "balanced_accuracy": compute_balanced_accuracy,
    "jaccard_index": compute_jaccard_index,
    # Advanced metrics
    "matthews_corrcoef": compute_matthews_corrcoef,
    "cohen_kappa": compute_cohen_kappa,
    "true_positives": compute_true_positives,
    "false_positives": compute_false_positives,
    "true_negatives": compute_true_negatives,
    "false_negatives": compute_false_negatives,
    "positive_predictive_value": compute_positive_predictive_value,
    "negative_predictive_value": compute_negative_predictive_value,
    "false_positive_rate": compute_false_positive_rate,
    "false_negative_rate": compute_false_negative_rate,
    "false_discovery_rate": compute_false_discovery_rate,
    "false_omission_rate": compute_false_omission_rate,
    "chi2_statistic": compute_chi2_statistic,
    "chi2_p_value": compute_chi2_p_value,
    "fisher_exact_p_value": compute_fisher_exact_p_value,
    # Probabilistic metrics
    "roc_auc": compute_roc_auc,
    "average_precision": compute_average_precision,
    "brier_score": compute_brier_score,
    "log_loss": compute_log_loss,
    # Curve metrics
    "roc_curve": compute_roc_curve,
    "pr_curve": compute_pr_curve,
    # Meteorological metrics
    "critical_success_index": compute_critical_success_index,
    "hit_rate": compute_hit_rate,
    "false_alarm_rate": compute_false_alarm_rate,
    "false_alarm_ratio": compute_false_alarm_ratio,
    "bias_score": compute_bias_score,
    "equitable_threat_score": compute_equitable_threat_score,
    "heidke_skill_score": compute_heidke_skill_score,
    # Legacy metrics
    "tpr": compute_tpr,
    "tnr": compute_tnr,
}


# ============================
# GROUPED COMPUTATION FUNCTIONS (for backwards compatibility)
# ============================


def prepare_arrays(preds, ys, probabilities=None):
    """
    Prepare prediction arrays for metric calculations.

    Args:
        preds (list or np.array): Predicted labels
        ys (list or np.array): True labels
        probabilities (list or np.array, optional): Predicted probabilities

    Returns:
        tuple: (preds_array, ys_array, probs_array_or_none)
    """
    if isinstance(preds, list):
        preds = np.concatenate(preds)
    if isinstance(ys, list):
        ys = np.concatenate(ys)

    probs = None
    if probabilities is not None:
        if isinstance(probabilities, list):
            probs = np.concatenate(probabilities)
        else:
            probs = probabilities

    return preds, ys, probs


def compute_basic_metrics(preds, ys):
    """
    Compute basic classification metrics using sklearn.

    Args:
        preds: Predicted labels
        ys: True labels

    Returns:
        dict: Dictionary of metric values
    """
    preds, ys, _ = prepare_arrays(preds, ys)

    return {
        "accuracy": accuracy_score(ys, preds),
        "precision": precision_score(ys, preds, zero_division=0),
        "recall": recall_score(ys, preds, zero_division=0),  # Same as TPR
        "f1_score": f1_score(ys, preds, zero_division=0),
        "specificity": _compute_tnr(preds, ys),  # TNR using helper function
        "balanced_accuracy": balanced_accuracy_score(ys, preds),
        "jaccard_index": jaccard_score(ys, preds, zero_division=0),
    }


def compute_advanced_metrics(preds, ys):
    """
    Compute advanced classification metrics.

    Args:
        preds: Predicted labels
        ys: True labels

    Returns:
        dict: Dictionary of metric values
    """
    preds, ys, _ = prepare_arrays(preds, ys)

    metrics = {
        "matthews_corrcoef": matthews_corrcoef(ys, preds),
        "cohen_kappa": cohen_kappa_score(ys, preds),
    }

    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(ys, preds, labels=[0, 1]).ravel()

    metrics.update(
        {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "positive_predictive_value": tp / (tp + fp)
            if (tp + fp) > 0
            else 0,  # Precision
            "negative_predictive_value": tn / (tn + fn) if (tn + fn) > 0 else 0,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "false_discovery_rate": fp / (fp + tp) if (fp + tp) > 0 else 0,
            "false_omission_rate": fn / (fn + tn) if (fn + tn) > 0 else 0,
        }
    )

    # Statistical significance tests
    try:
        contingency_table = confusion_matrix(ys, preds, labels=[0, 1])
        chi2_stat, chi2_p_value, _, _ = chi2_contingency(contingency_table)
        metrics["chi2_statistic"] = chi2_stat
        metrics["chi2_p_value"] = chi2_p_value

        # Fisher's exact test for small samples
        if np.sum(contingency_table) < 1000:  # Use Fisher's exact for smaller samples
            _, fisher_p_value = fisher_exact(contingency_table)
            metrics["fisher_exact_p_value"] = fisher_p_value

    except (ValueError, ZeroDivisionError):
        metrics["chi2_statistic"] = np.nan
        metrics["chi2_p_value"] = np.nan
        metrics["fisher_exact_p_value"] = np.nan

    return metrics


def compute_probabilistic_metrics(probabilities, ys):
    """
    Compute metrics that require probability estimates.

    Args:
        probabilities: Predicted probabilities for positive class
        ys: True labels

    Returns:
        dict: Dictionary of metric values
    """
    _, ys, probs = prepare_arrays(None, ys, probabilities)

    if probs is None:
        return {}

    try:
        metrics = {
            "roc_auc": roc_auc_score(ys, probs),
            "average_precision": average_precision_score(ys, probs),
            "brier_score": brier_score_loss(ys, probs),
        }

        # Convert probabilities to binary predictions for log loss
        # Avoid log(0) by clipping probabilities
        probs_clipped = np.clip(probs, 1e-15, 1 - 1e-15)
        metrics["log_loss"] = log_loss(ys, probs_clipped)

    except (ValueError, TypeError) as e:
        warnings.warn(f"Error computing probabilistic metrics: {e}")
        metrics = {
            "roc_auc": np.nan,
            "average_precision": np.nan,
            "brier_score": np.nan,
            "log_loss": np.nan,
        }

    return metrics


def compute_threshold_metrics(probabilities, ys, thresholds=None):
    """
    Compute metrics at different thresholds for ROC and PR curves.

    Args:
        probabilities: Predicted probabilities
        ys: True labels
        thresholds: Custom thresholds (optional)

    Returns:
        dict: ROC and PR curve data
    """
    _, ys, probs = prepare_arrays(None, ys, probabilities)

    if probs is None:
        return {}

    try:
        # ROC curve
        fpr, tpr_vals, roc_thresholds = roc_curve(ys, probs)

        # Precision-Recall curve
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(ys, probs)

        return {
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr_vals.tolist(),
                "thresholds": roc_thresholds.tolist(),
            },
            "pr_curve": {
                "precision": precision_vals.tolist(),
                "recall": recall_vals.tolist(),
                "thresholds": pr_thresholds.tolist(),
            },
        }
    except (ValueError, TypeError):
        return {}


def compute_all_metrics(preds, ys, probabilities=None, include_curves=False):
    """
    Compute all available metrics for binary classification.

    Args:
        preds: Predicted labels
        ys: True labels
        probabilities: Predicted probabilities (optional)
        include_curves: Whether to include ROC/PR curve data

    Returns:
        dict: Comprehensive metrics dictionary
    """
    # Combine all metrics
    metrics = {}

    # Basic metrics
    metrics.update(compute_basic_metrics(preds, ys))

    # Advanced metrics
    metrics.update(compute_advanced_metrics(preds, ys))

    # Probabilistic metrics (if probabilities provided)
    if probabilities is not None:
        metrics.update(compute_probabilistic_metrics(probabilities, ys))

        # Curve data (if requested)
        if include_curves:
            metrics.update(compute_threshold_metrics(probabilities, ys))

    return metrics


def get_classification_report(preds, ys, target_names=None):
    """
    Generate a detailed classification report.

    Args:
        preds: Predicted labels
        ys: True labels
        target_names: Names for the classes

    Returns:
        str: Formatted classification report
    """
    preds, ys, _ = prepare_arrays(preds, ys)

    if target_names is None:
        target_names = ["Dry", "Wet"]  # For wet/dry classification

    return classification_report(ys, preds, target_names=target_names)


# ============================
# METEOROLOGICAL SPECIFIC METRICS
# ============================


def compute_meteorological_metrics(preds, ys, probabilities=None):
    """
    Compute metrics specifically relevant for meteorological/hydrological applications.

    Args:
        preds: Predicted labels
        ys: True labels
        probabilities: Predicted probabilities (optional)

    Returns:
        dict: Meteorological metrics
    """
    preds, ys, _ = prepare_arrays(preds, ys)

    # Get confusion matrix components
    tn, fp, fn, tp = confusion_matrix(ys, preds, labels=[0, 1]).ravel()

    metrics = {
        # Critical Success Index (Threat Score) - important for precipitation
        "critical_success_index": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0,
        # Hit Rate (Probability of Detection) - same as recall/sensitivity
        "hit_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
        # False Alarm Rate
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        # False Alarm Ratio
        "false_alarm_ratio": fp / (fp + tp) if (fp + tp) > 0 else 0,
        # Bias Score (frequency bias)
        "bias_score": (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0,
        # Equitable Threat Score
        "equitable_threat_score": _compute_equitable_threat_score(tp, fp, fn, tn),
        # Heidke Skill Score (similar to Cohen's kappa)
        "heidke_skill_score": _compute_heidke_skill_score(tp, fp, fn, tn),
    }

    return metrics


def _compute_equitable_threat_score(tp, fp, fn, tn):
    """Compute Equitable Threat Score."""
    n = tp + fp + fn + tn
    hits_random = ((tp + fp) * (tp + fn)) / n if n > 0 else 0
    return (
        (tp - hits_random) / (tp + fp + fn - hits_random)
        if (tp + fp + fn - hits_random) != 0
        else 0
    )


def _compute_heidke_skill_score(tp, fp, fn, tn):
    """Compute Heidke Skill Score."""
    n = tp + fp + fn + tn
    po = (tp + tn) / n if n > 0 else 0  # Observed accuracy
    pe = (
        (((tp + fn) * (tp + fp)) + ((tn + fp) * (tn + fn))) / (n * n) if n > 0 else 0
    )  # Expected accuracy
    return (po - pe) / (1 - pe) if pe != 1 else 0


# ============================
# METRICS CALCULATOR CLASS
# ============================


class MetricsCalculator:
    """
    A comprehensive metrics calculator for binary classification with configurable metric selection.

    This class provides a unified interface to compute any subset of available metrics
    for wet/dry binary classification tasks using sklearn, scipy, and meteorological metrics.
    """

    # Define all available metrics organized by category
    BASIC_METRICS = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "balanced_accuracy",
        "jaccard_index",
    ]

    ADVANCED_METRICS = [
        "matthews_corrcoef",
        "cohen_kappa",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "positive_predictive_value",
        "negative_predictive_value",
        "false_positive_rate",
        "false_negative_rate",
        "false_discovery_rate",
        "false_omission_rate",
        "chi2_statistic",
        "chi2_p_value",
        "fisher_exact_p_value",
    ]

    PROBABILISTIC_METRICS = ["roc_auc", "average_precision", "brier_score", "log_loss"]

    METEOROLOGICAL_METRICS = [
        "critical_success_index",
        "hit_rate",
        "false_alarm_rate",
        "false_alarm_ratio",
        "bias_score",
        "equitable_threat_score",
        "heidke_skill_score",
    ]

    CURVE_METRICS = ["roc_curve", "pr_curve"]

    # Legacy metrics (for backwards compatibility)
    LEGACY_METRICS = ["tpr", "tnr"]

    def __init__(self, metrics=None, include_curves=False):
        """
        Initialize the MetricsCalculator with specified metrics.

        Args:
            metrics (list, optional): List of metric names to compute. If None, computes all basic metrics.
            include_curves (bool): Whether to include ROC and PR curve data.
        """
        self.include_curves = include_curves

        if metrics is None:
            # Default to basic metrics
            self.requested_metrics = self.BASIC_METRICS.copy()
        else:
            # Validate requested metrics
            available_metrics = self.get_available_metrics()
            invalid_metrics = set(metrics) - set(available_metrics)
            if invalid_metrics:
                raise ValueError(f"Invalid metrics requested: {invalid_metrics}")
            self.requested_metrics = list(metrics)

        # Add curves if requested
        if include_curves:
            self.requested_metrics.extend(self.CURVE_METRICS)

    @classmethod
    def get_available_metrics(cls):
        """
        Get a list of all available metric names.

        Returns:
            list: List of all available metric names
        """
        return (
            cls.BASIC_METRICS
            + cls.ADVANCED_METRICS
            + cls.PROBABILISTIC_METRICS
            + cls.METEOROLOGICAL_METRICS
            + cls.CURVE_METRICS
            + cls.LEGACY_METRICS
        )

    @classmethod
    def get_metrics_by_category(cls):
        """
        Get metrics organized by category.

        Returns:
            dict: Dictionary with categories as keys and metric lists as values
        """
        return {
            "basic": cls.BASIC_METRICS,
            "advanced": cls.ADVANCED_METRICS,
            "probabilistic": cls.PROBABILISTIC_METRICS,
            "meteorological": cls.METEOROLOGICAL_METRICS,
            "curves": cls.CURVE_METRICS,
            "legacy": cls.LEGACY_METRICS,
        }

    def requires_probabilities(self):
        """
        Check if any requested metrics require probability estimates.

        Returns:
            bool: True if probabilities are required
        """
        prob_required_metrics = set(self.PROBABILISTIC_METRICS + self.CURVE_METRICS)
        return bool(set(self.requested_metrics) & prob_required_metrics)

    def compute_metrics(self, preds, ys, probabilities=None):
        """
        Compute the requested metrics efficiently using individual metric functions.

        Args:
            preds: Predicted labels
            ys: True labels
            probabilities: Predicted probabilities (required for probabilistic metrics)

        Returns:
            dict: Dictionary containing only the requested metrics
        """
        # Clear cache before computing to ensure fresh computation
        global _confusion_cache
        _confusion_cache.clear()

        # Check if probabilities are needed but not provided
        if self.requires_probabilities() and probabilities is None:
            raise ValueError(
                "Probabilities are required for the requested metrics but were not provided"
            )

        # Compute only the requested metrics efficiently
        computed_metrics = {}

        for metric_name in self.requested_metrics:
            if metric_name in METRIC_FUNCTIONS:
                try:
                    computed_metrics[metric_name] = METRIC_FUNCTIONS[metric_name](
                        preds, ys, probabilities
                    )
                except Exception as e:
                    warnings.warn(f"Error computing {metric_name}: {e}")
                    computed_metrics[metric_name] = np.nan

        return computed_metrics

    def compute_summary_report(self, preds, ys, probabilities=None):
        """
        Compute metrics and return a formatted summary report.

        Args:
            preds: Predicted labels
            ys: True labels
            probabilities: Predicted probabilities (optional)

        Returns:
            str: Formatted summary report
        """
        metrics = self.compute_metrics(preds, ys, probabilities)

        report = []
        report.append("=" * 60)
        report.append("BINARY CLASSIFICATION METRICS SUMMARY")
        report.append("=" * 60)

        # Group metrics by category for better readability
        categories = self.get_metrics_by_category()

        for category, metric_names in categories.items():
            category_metrics = {k: v for k, v in metrics.items() if k in metric_names}
            if category_metrics:
                report.append(f"\n{category.upper()} METRICS:")
                report.append("-" * 30)
                for metric_name, value in category_metrics.items():
                    if isinstance(value, dict):  # For curve data
                        report.append(f"{metric_name}: [curve data available]")
                    elif isinstance(value, (int, float)):
                        if np.isnan(value):
                            report.append(f"{metric_name}: N/A")
                        else:
                            report.append(f"{metric_name}: {value:.4f}")
                    else:
                        report.append(f"{metric_name}: {value}")

        # Add classification report if basic metrics are included
        if any(
            metric in self.requested_metrics
            for metric in ["precision", "recall", "f1_score"]
        ):
            report.append("\nDETAILED CLASSIFICATION REPORT:")
            report.append("-" * 40)
            report.append(get_classification_report(preds, ys))

        return "\n".join(report)

    def __repr__(self):
        """String representation of the MetricsCalculator."""
        return (
            f"MetricsCalculator(metrics={self.requested_metrics}, "
            f"include_curves={self.include_curves})"
        )


# ============================
# CONVENIENCE FUNCTIONS
# ============================


def compute_single_metric(metric_name, preds, ys, probabilities=None):
    """
    Compute a single metric by name.

    Args:
        metric_name: Name of the metric to compute
        preds: Predicted labels
        ys: True labels
        probabilities: Predicted probabilities (optional)

    Returns:
        Computed metric value
    """
    if metric_name not in METRIC_FUNCTIONS:
        raise ValueError(
            f"Unknown metric: {metric_name}. Available: {list(METRIC_FUNCTIONS.keys())}"
        )

    # Clear cache for fresh computation
    global _confusion_cache
    _confusion_cache.clear()

    return METRIC_FUNCTIONS[metric_name](preds, ys, probabilities)


def quick_metrics(preds, ys, probabilities=None, metric_set="basic"):
    """
    Quick calculation of common metric sets.

    Args:
        preds: Predicted labels
        ys: True labels
        probabilities: Predicted probabilities (optional)
        metric_set: 'basic', 'advanced', 'meteorological', or 'all'

    Returns:
        dict: Computed metrics
    """
    if metric_set == "basic":
        calculator = MetricsCalculator(metrics=MetricsCalculator.BASIC_METRICS)
    elif metric_set == "advanced":
        calculator = MetricsCalculator(
            metrics=MetricsCalculator.BASIC_METRICS + MetricsCalculator.ADVANCED_METRICS
        )
    elif metric_set == "meteorological":
        calculator = MetricsCalculator(
            metrics=MetricsCalculator.BASIC_METRICS
            + MetricsCalculator.METEOROLOGICAL_METRICS
        )
    elif metric_set == "all":
        all_metrics = (
            MetricsCalculator.BASIC_METRICS
            + MetricsCalculator.ADVANCED_METRICS
            + MetricsCalculator.METEOROLOGICAL_METRICS
        )
        if probabilities is not None:
            all_metrics += MetricsCalculator.PROBABILISTIC_METRICS
        calculator = MetricsCalculator(metrics=all_metrics)
    else:
        raise ValueError(f"Invalid metric_set: {metric_set}")

    return calculator.compute_metrics(preds, ys, probabilities)
