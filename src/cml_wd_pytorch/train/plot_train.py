import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves_from_csv(csv_path, save_dir, filename="training_curves.png"):
    """
    Create a flexible grid plot of training curves from a CSV file.

    Automatically detects train/val metric pairs and creates one panel per metric
    with epochs on x-axis and both train/val versions in the same panel.

    Args:
        csv_path (str): Path to the CSV file containing training metrics
        save_dir (str): Directory where to save the plot
        filename (str): Name of the output plot file

    Returns:
        str: Path to the saved plot file
    """
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file {csv_path} does not exist. Skipping plot generation.")
        return None

    # Read the CSV
    df = pd.read_csv(csv_path)

    if "epoch" not in df.columns:
        print(
            f"Warning: No 'epoch' column found in {csv_path}. Skipping plot generation."
        )
        return None

    epochs = df["epoch"]

    # Find all train/val metric pairs
    metric_pairs = {}
    train_cols = [col for col in df.columns if col.startswith("train_")]

    for train_col in train_cols:
        metric_name = train_col.replace("train_", "")
        val_col = f"val_{metric_name}"

        if val_col in df.columns:
            metric_pairs[metric_name] = {"train": train_col, "val": val_col}

    if not metric_pairs:
        print(
            f"Warning: No train/val metric pairs found in {csv_path}. Skipping plot generation."
        )
        return None

    # Calculate grid dimensions
    n_metrics = len(metric_pairs)
    n_cols = min(3, n_metrics)  # Max 3 columns
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division

    # Create the plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Handle case where we have only one subplot
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1 and n_cols > 1:
        axes = axes.flatten()
    elif n_rows > 1:
        axes = axes.flatten()

    # Plot each metric
    for idx, (metric_name, cols) in enumerate(metric_pairs.items()):
        ax = axes[idx] if n_metrics > 1 else axes[0]

        train_values = df[cols["train"]]
        val_values = df[cols["val"]]

        # Plot train and validation curves
        ax.plot(
            epochs,
            train_values,
            label=f"Train {metric_name}",
            color="blue",
            linewidth=2,
        )
        ax.plot(
            epochs, val_values, label=f"Val {metric_name}", color="orange", linewidth=2
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"{metric_name.upper()} Training Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set reasonable y-limits based on metric type
        if "bce" in metric_name or "loss" in metric_name:
            ax.set_ylim(bottom=0)
        elif any(
            name in metric_name for name in ["acc", "precision", "recall", "f1", "auc"]
        ):
            ax.set_ylim(0, 1)

    # Hide empty subplots
    if n_metrics < len(axes):
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

    plt.tight_layout()

    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Training curves plot saved to: {save_path}")
    return save_path


def plot_training_history(loss_dict, run_id, package_path):
    """
    Legacy function for backward compatibility.
    Plot training history from loss dictionary and save the figure.

    Args:
        loss_dict (dict): Dictionary containing training history with keys like 'epoch', 'train_bce', 'val_bce', etc.
        run_id (str): Unique identifier for the training run.
        package_path (str): Base path of the package to construct the full save path.
    """

    # plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(loss_dict["train_bce"], label="Train BCE", color="blue")
    plt.plot(loss_dict["val_bce"], label="Validation BCE", color="orange")

    # Add other metrics if they exist
    if "train_acc" in loss_dict:
        plt.plot(loss_dict["train_acc"], label="Train ACC", color="green")
        plt.plot(loss_dict["val_acc"], label="Validation ACC", color="red")

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True)
    save_path = str(package_path) + "/results/%s/plots/loss_curves.png" % run_id
    plt.savefig(save_path)
    plt.close()

    print("Training history plot saved to: ", save_path)

    return None
