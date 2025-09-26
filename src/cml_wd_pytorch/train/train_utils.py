"""
Training utilities for the CML wet/dry PyTorch project.
"""


class EarlyStopping:
    """
    Minimalistic early stopping implementation.

    Monitors a metric and stops training when it stops improving for a specified patience.
    """

    def __init__(self, patience=7, min_delta=0.0, mode="min"):
        """
        Initialize early stopping.

        Args:
            patience (int): Number of epochs to wait after last improvement before stopping.
            min_delta (float): Minimum change to qualify as an improvement.
            mode (str): 'min' for metrics that should decrease (loss), 'max' for metrics that should increase (accuracy).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

        # Set comparison function based on mode
        if mode == "min":
            self.is_better = lambda current, best: current < best - min_delta
        elif mode == "max":
            self.is_better = lambda current, best: current > best + min_delta
        else:
            raise ValueError("Mode must be 'min' or 'max'")

    def __call__(self, metric, epoch=None):
        """
        Check if training should be stopped.

        Args:
            metric (float): Current metric value to monitor.
            epoch (int, optional): Current epoch number.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if self.best_score is None:
            self.best_score = metric
            self.best_epoch = epoch
        elif self.is_better(metric, self.best_score):
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False


class MetricTracker:
    """
    Class to track and store training and validation metrics over epochs.
    """

    def __init__(self, metrics=None):
        """
        Initialize metric tracker.

        Args:
            metrics (list): List of metric names to track.
        """
        self.metrics = {
            "epoch": [],
        }
        if metrics is not None:
            for metric in metrics:
                self.metrics[f"train_{metric}"] = []
                self.metrics[f"val_{metric}"] = []
        else:
            raise ValueError("Please provide a list of metrics to track.")

        # Temporary storage for current epoch metrics
        self.current_epoch_metrics = {
            "train": {metric: [] for metric in metrics},
            "val": {metric: [] for metric in metrics},
        }

    def append(self, value, metric_name="bce", phase="train"):
        """
        Append a metric value for the current epoch.

        Args:
            value (float): Metric value to append.
            metric_name (str): Name of the metric.
            phase (str): 'train' or 'val'.
        """
        if metric_name in self.current_epoch_metrics[phase]:
            self.current_epoch_metrics[phase][metric_name].append(value)

    def log_epoch(self, epoch, train_metrics=None, val_metrics=None):
        """
        Log metrics for a given epoch.

        Args:
            epoch (int): Current epoch number.
            train_metrics (dict, optional): Dictionary of training metrics.
            val_metrics (dict, optional): Dictionary of validation metrics.
        """
        self.metrics["epoch"].append(epoch)

        # Compute accumulated metrics first
        accumulated_train = {}
        for metric, values in self.current_epoch_metrics["train"].items():
            if values:
                accumulated_train[metric] = sum(values) / len(values)

        accumulated_val = {}
        for metric, values in self.current_epoch_metrics["val"].items():
            if values:
                accumulated_val[metric] = sum(values) / len(values)

        # Merge accumulated and provided metrics (provided metrics take precedence)
        final_train_metrics = accumulated_train.copy()
        if train_metrics:
            final_train_metrics.update(train_metrics)

        final_val_metrics = accumulated_val.copy()
        if val_metrics:
            final_val_metrics.update(val_metrics)

        # Store epoch metrics
        for key, value in final_train_metrics.items():
            if f"train_{key}" in self.metrics:
                self.metrics[f"train_{key}"].append(value)

        for key, value in final_val_metrics.items():
            if f"val_{key}" in self.metrics:
                self.metrics[f"val_{key}"].append(value)

    def reset(self):
        """Reset current epoch metrics."""
        for phase in self.current_epoch_metrics:
            for metric in self.current_epoch_metrics[phase]:
                self.current_epoch_metrics[phase][metric] = []

    def get_metrics(self):
        """Return the stored metrics."""
        return self.metrics

    def get_latest(self, metric_name, phase="val"):
        """
        Get the latest value for a specific metric.

        Args:
            metric_name (str): Name of the metric.
            phase (str): 'train' or 'val'.

        Returns:
            float: Latest metric value.
        """
        key = f"{phase}_{metric_name}"
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return None
