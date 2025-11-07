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


# ============================
# INDIVIDUAL METRIC FUNCTIONS
# ============================


def compute_accuracy(preds, ys, probabilities=None):
    """Compute accuracy score.

    Measures the fraction of predictions that match the true labels (correct predictions / total predictions).
    """
    preds, ys, _ = prepare_arrays(preds, ys)
    return accuracy_score(ys, preds)


def compute_precision(preds, ys, probabilities=None):
    """Compute precision score.

    Measures the fraction of positive predictions that are actually positive (TP / (TP + FP)).
    """
    preds, ys, _ = prepare_arrays(preds, ys)
    return precision_score(ys, preds, zero_division=0)


def compute_recall(preds, ys, probabilities=None):
    """Compute recall score (same as TPR).

    Measures the fraction of actual positives that were correctly predicted (TP / (TP + FN)).
    """
    preds, ys, _ = prepare_arrays(preds, ys)
    return recall_score(ys, preds, zero_division=0)


def compute_f1_score(preds, ys, probabilities=None):
    """Compute F1 score.

    Harmonic mean of precision and recall, balancing both metrics (2 * precision * recall / (precision + recall)).
    """
    preds, ys, _ = prepare_arrays(preds, ys)
    return f1_score(ys, preds, zero_division=0)


def compute_specificity(preds, ys, probabilities=None):
    """Compute specificity (TNR).

    Measures the fraction of actual negatives that were correctly predicted (TN / (TN + FP)).
    """
    return _compute_tnr(preds, ys)


def compute_balanced_accuracy(preds, ys, probabilities=None):
    """Compute balanced accuracy.

    Average of recall for each class, adjusting for class imbalance ((sensitivity + specificity) / 2).
    """
    preds, ys, _ = prepare_arrays(preds, ys)
    return balanced_accuracy_score(ys, preds)


def compute_jaccard_index(preds, ys, probabilities=None):
    """Compute Jaccard index (IoU).

    Measures similarity between predicted and true sets, intersection over union (TP / (TP + FP + FN)).
    """
    preds, ys, _ = prepare_arrays(preds, ys)
    return jaccard_score(ys, preds, zero_division=0)


def compute_matthews_corrcoef(preds, ys, probabilities=None):
    """Compute Matthews correlation coefficient.

    Correlation coefficient between observed and predicted classifications, ranging from -1 to 1.
    """
    preds, ys, _ = prepare_arrays(preds, ys)
    return matthews_corrcoef(ys, preds)


def compute_cohen_kappa(preds, ys, probabilities=None):
    """Compute Cohen's kappa.

    Measures inter-rater agreement, accounting for agreement occurring by chance (0 = chance, 1 = perfect).
    """
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
    """Compute true positives count.

    Number of correctly predicted positive instances (wet conditions correctly identified as wet).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return tp


def compute_false_positives(preds, ys, probabilities=None):
    """Compute false positives count.

    Number of incorrectly predicted positive instances (dry conditions incorrectly identified as wet).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fp


def compute_true_negatives(preds, ys, probabilities=None):
    """Compute true negatives count.

    Number of correctly predicted negative instances (dry conditions correctly identified as dry).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return tn


def compute_false_negatives(preds, ys, probabilities=None):
    """Compute false negatives count.

    Number of incorrectly predicted negative instances (wet conditions incorrectly identified as dry).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fn


def compute_positive_predictive_value(preds, ys, probabilities=None):
    """Compute positive predictive value (precision).

    Proportion of predicted positive instances that are actually positive (TP / (TP + FP)).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def compute_negative_predictive_value(preds, ys, probabilities=None):
    """Compute negative predictive value.

    Proportion of predicted negative instances that are actually negative (TN / (TN + FN)).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return tn / (tn + fn) if (tn + fn) > 0 else 0


def compute_false_positive_rate(preds, ys, probabilities=None):
    """Compute false positive rate.

    Proportion of actual negative instances incorrectly classified as positive (FP / (FP + TN)).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def compute_false_negative_rate(preds, ys, probabilities=None):
    """Compute false negative rate.

    Proportion of actual positive instances incorrectly classified as negative (FN / (FN + TP)).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fn / (fn + tp) if (fn + tp) > 0 else 0


def compute_false_discovery_rate(preds, ys, probabilities=None):
    """Compute false discovery rate.

    Proportion of predicted positive instances that are actually negative (FP / (FP + TP)).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fp / (fp + tp) if (fp + tp) > 0 else 0


def compute_false_omission_rate(preds, ys, probabilities=None):
    """Compute false omission rate.

    Proportion of predicted negative instances that are actually positive (FN / (FN + TN)).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fn / (fn + tn) if (fn + tn) > 0 else 0


def compute_chi2_test(preds, ys, probabilities=None):
    """Compute chi-square test statistics.

    Tests independence between predicted and actual classifications using chi-square test.
    """
    preds, ys, _ = prepare_arrays(preds, ys)
    try:
        contingency_table = confusion_matrix(ys, preds, labels=[0, 1])
        chi2_stat, chi2_p_value, _, _ = chi2_contingency(contingency_table)
        return {"chi2_statistic": chi2_stat, "chi2_p_value": chi2_p_value}
    except (ValueError, ZeroDivisionError):
        return {"chi2_statistic": np.nan, "chi2_p_value": np.nan}


def compute_chi2_statistic(preds, ys, probabilities=None):
    """Compute chi-square statistic.

    Chi-square test statistic measuring association between predicted and actual classifications.
    """
    return compute_chi2_test(preds, ys)["chi2_statistic"]


def compute_chi2_p_value(preds, ys, probabilities=None):
    """Compute chi-square p-value.

    P-value from chi-square test indicating significance of association between predictions and truth.
    """
    return compute_chi2_test(preds, ys)["chi2_p_value"]


def compute_fisher_exact_p_value(preds, ys, probabilities=None):
    """Compute Fisher's exact test p-value.

    Exact test for independence in 2x2 contingency tables, more accurate than chi-square for small samples.
    """
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
    """Compute ROC AUC score.

    Area under the Receiver Operating Characteristic curve, measures model's ability to distinguish classes.
    """
    if probabilities is None:
        raise ValueError("Probabilities required for ROC AUC")
    _, ys, probs = prepare_arrays(preds, ys, probabilities)
    try:
        return roc_auc_score(ys, probs)
    except (ValueError, TypeError):
        warnings.warn("Error computing ROC AUC")
        return np.nan


def compute_average_precision(preds, ys, probabilities=None):
    """Compute average precision (PR AUC).

    Area under the Precision-Recall curve, measures performance across all probability thresholds.
    """
    if probabilities is None:
        raise ValueError("Probabilities required for average precision")
    _, ys, probs = prepare_arrays(preds, ys, probabilities)
    try:
        return average_precision_score(ys, probs)
    except (ValueError, TypeError):
        warnings.warn("Error computing average precision")
        return np.nan


def compute_brier_score(preds, ys, probabilities=None):
    """Compute Brier score.

    Mean squared difference between predicted probabilities and actual outcomes (lower is better).
    """
    if probabilities is None:
        raise ValueError("Probabilities required for Brier score")
    _, ys, probs = prepare_arrays(preds, ys, probabilities)
    try:
        return brier_score_loss(ys, probs)
    except (ValueError, TypeError):
        warnings.warn("Error computing Brier score")
        return np.nan


def compute_log_loss(preds, ys, probabilities=None):
    """Compute log loss.

    Logarithmic loss measuring probability calibration quality (lower is better, penalizes confident wrong predictions).
    """
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
    """Compute ROC curve data.

    Returns ROC curve coordinates (FPR, TPR) across all probability thresholds for visualization.
    """
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
    """Compute Precision-Recall curve data.

    Returns Precision-Recall curve coordinates across all probability thresholds for visualization.
    """
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
    """Compute Critical Success Index (Threat Score).

    Meteorological metric measuring forecast skill for binary events (TP / (TP + FP + FN)).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0


def compute_hit_rate(preds, ys, probabilities=None):
    """Compute Hit Rate (Probability of Detection).

    Proportion of observed events that were correctly predicted (TP / (TP + FN)).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def compute_false_alarm_rate(preds, ys, probabilities=None):
    """Compute False Alarm Rate.

    Proportion of non-events that were incorrectly predicted as events (FP / (FP + TN)).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def compute_false_alarm_ratio(preds, ys, probabilities=None):
    """Compute False Alarm Ratio.

    Proportion of predicted events that did not occur (FP / (FP + TP)).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return fp / (fp + tp) if (fp + tp) > 0 else 0


def compute_bias_score(preds, ys, probabilities=None):
    """Compute Bias Score (frequency bias).

    Ratio of predicted to observed event frequencies, indicates over/under-forecasting (1 = perfect).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0


def compute_equitable_threat_score(preds, ys, probabilities=None):
    """Compute Equitable Threat Score.

    Threat score adjusted for hits expected by chance, ranges from -1/3 to 1 (0 = no skill).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return _compute_equitable_threat_score(tp, fp, fn, tn)


def compute_heidke_skill_score(preds, ys, probabilities=None):
    """Compute Heidke Skill Score.

    Accuracy adjusted for chance agreement, ranges from -1 to 1 (0 = no skill, 1 = perfect).
    """
    tn, fp, fn, tp = _get_confusion_matrix_components(preds, ys)
    return _compute_heidke_skill_score(tp, fp, fn, tn)


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
}


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
            report.append(classification_report(ys, preds, target_names=["Dry", "Wet"]))

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
