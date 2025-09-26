"""
Best Model Logger for tracking and saving the best performing models during training.

This module encapsulates the logic for:
- Tracking best model performance
- Computing comprehensive metrics for the best model
- Saving model checkpoints and metadata
- Managing file cleanup for previous best models
"""

import json
import os
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cml_wd_pytorch.evaluation.scores import MetricsCalculator

from .plot_train import plot_training_curves_from_csv


class BestModelLogger:
    """
    Manages best model tracking, comprehensive metrics computation, and file logging.

    This class encapsulates all the logic for determining when a new best model is found,
    computing comprehensive evaluation metrics, and saving model checkpoints with metadata.
    """

    def __init__(
        self, package_path: str, run_id: str, comprehensive_metrics: MetricsCalculator
    ):
        """
        Initialize the BestModelLogger.

        Args:
            package_path: Root path of the package
            run_id: Unique identifier for this training run
            comprehensive_metrics: MetricsCalculator instance for computing detailed metrics
        """
        self.package_path = package_path
        self.run_id = run_id
        self.comprehensive_metrics = comprehensive_metrics

        # Track best model state
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.best_model_path: Optional[str] = None

        # Track essential metrics across epochs
        self.essential_metrics_history = {"epoch": [], "train_bce": [], "val_bce": []}

    def log_losses(self, epoch: int, train_bce: float, val_bce: float):
        """
        Log training and validation losses and save to CSV.

        Args:
            epoch: Current training epoch
            train_bce: Training BCE loss
            val_bce: Validation BCE loss
        """
        # Update metrics history
        self.essential_metrics_history["epoch"].append(epoch)
        self.essential_metrics_history["train_bce"].append(train_bce)
        self.essential_metrics_history["val_bce"].append(val_bce)

        # Save to CSV file
        df_essential = pd.DataFrame(self.essential_metrics_history)
        essential_metrics_path = (
            f"{self.package_path}/results/{self.run_id}/scores/scores_essential.csv"
        )
        df_essential.to_csv(essential_metrics_path, index=False)

        print(f"Essential scores saved to: {essential_metrics_path}")

    def check_and_update_best_model(
        self,
        model: torch.nn.Module,
        dataloader_val,
        process_batch_fn,
        device: torch.device,
        config: Dict[str, Any],
        epoch: int,
        current_val_loss: float,
        train_bce: float,
    ) -> bool:
        """
        Check if current model is the best so far and update if necessary.
        Also logs essential metrics for every epoch.

        Args:
            model: PyTorch model to evaluate
            dataloader_val: Validation dataloader
            process_batch_fn: Function to process batches (e.g., process_batch)
            device: Device to run computation on
            config: Configuration dictionary
            epoch: Current training epoch
            current_val_loss: Current validation loss
            train_bce: Training BCE loss

        Returns:
            bool: True if this was a new best model, False otherwise
        """
        # Always log essential metrics for every epoch
        self.log_losses(epoch, train_bce, current_val_loss)

        if current_val_loss < self.best_val_loss:
            print(
                f"\n🎯 NEW BEST MODEL! Validation loss improved from {self.best_val_loss:.4f} to {current_val_loss:.4f}"
            )

            # Remove previous best model if it exists
            if self.best_model_path and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
                print(f"Removed previous best model: {self.best_model_path}")

                # Also remove JIT model and config if they exist
                jit_path = self.best_model_path.replace(
                    "best_model.pth", "best_model_jit.pt"
                )
                config_path = jit_path.replace(".pt", "_config.json")

                if os.path.exists(jit_path):
                    os.remove(jit_path)
                    print(f"Removed previous JIT model: {jit_path}")
                if os.path.exists(config_path):
                    os.remove(config_path)
                    print(f"Removed previous config: {config_path}")

            # Update best metrics
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch
            self.best_model_path = (
                f"{self.package_path}/results/{self.run_id}/models/best_model.pth"
            )

            # Save new best model (traditional format)
            torch.save(model.state_dict(), self.best_model_path)
            print(f"New best model saved to: {self.best_model_path}")

            # Save self-contained JIT scripted model (for production)
            jit_model_path = (
                f"{self.package_path}/results/{self.run_id}/models/best_model_jit.pt"
            )
            try:
                # Create JIT scripted model
                model.eval()  # Set to eval mode for scripting
                scripted_model = torch.jit.script(model)
                torch.jit.save(scripted_model, jit_model_path)
                print(f"✅ JIT scripted model saved to: {jit_model_path}")
                print("  ↳ Use for production deployment (no CNN import needed)")

                # Save config alongside the JIT model
                config_path = jit_model_path.replace(".pt", "_config.json")
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2, default=str)
                print(f"✅ Model config saved to: {config_path}")

            except Exception as e:
                print(f"⚠️  Warning: Could not create JIT scripted model: {e}")
                print("   Continuing with standard model saving (development use)...")

            # Compute and save comprehensive metrics
            self._compute_and_save_comprehensive_metrics(
                model,
                dataloader_val,
                process_batch_fn,
                device,
                config,
                epoch,
                current_val_loss,
            )

            # Generate training curve plots
            self.generate_training_plots()

            print("=" * 80)
            return True
        else:
            print(
                f"No improvement. Best validation loss remains {self.best_val_loss:.4f} (epoch {self.best_epoch})"
            )
            return False

    def _compute_and_save_comprehensive_metrics(
        self,
        model: torch.nn.Module,
        dataloader_val,
        process_batch_fn,
        device: torch.device,
        config: Dict[str, Any],
        epoch: int,
        current_val_loss: float,
    ):
        """
        Compute comprehensive metrics for the best model and save reports.

        Args:
            model: PyTorch model to evaluate
            dataloader_val: Validation dataloader
            process_batch_fn: Function to process batches
            device: Device to run computation on
            config: Configuration dictionary
            epoch: Current training epoch
            current_val_loss: Current validation loss
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE METRICS SUMMARY FOR BEST EPOCH")
        print("=" * 80)

        # Compute comprehensive metrics with all predictions
        val_probs, val_preds_comprehensive, val_ys_comprehensive = (
            self._collect_predictions(
                model, dataloader_val, process_batch_fn, device, config
            )
        )

        # Generate comprehensive metrics report
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*ill-defined.*")
            warnings.filterwarnings("ignore", message=".*invalid value encountered.*")

            comprehensive_report = self.comprehensive_metrics.compute_summary_report(
                val_preds_comprehensive,
                val_ys_comprehensive,
                probabilities=val_probs,
            )

        print(comprehensive_report)

        # Save comprehensive metrics report to file
        report_path = f"{self.package_path}/results/{self.run_id}/scores/best_epoch_comprehensive_metrics.txt"
        with open(report_path, "w") as f:
            f.write(f"Best Epoch: {epoch}\n")
            f.write(f"Best Validation Loss: {current_val_loss:.6f}\n\n")
            f.write(comprehensive_report)

        print(f"\nComprehensive metrics saved to: {report_path}")

        # Save metadata about the best model
        self._save_model_metadata(epoch, current_val_loss, report_path)

    def _collect_predictions(
        self,
        model: torch.nn.Module,
        dataloader_val,
        process_batch_fn,
        device: torch.device,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect predictions and ground truth labels from the validation set.

        Args:
            model: PyTorch model to evaluate
            dataloader_val: Validation dataloader
            process_batch_fn: Function to process batches
            device: Device to run computation on
            config: Configuration dictionary

        Returns:
            Tuple of (probabilities, predictions, ground_truth_labels)
        """
        val_probs = []
        val_preds_comprehensive = []
        val_ys_comprehensive = []

        model.eval()
        with torch.no_grad():
            for i, batch in tqdm(
                enumerate(dataloader_val), desc="Computing comprehensive metrics"
            ):
                x, y = process_batch_fn(batch, device, config["data"]["reflength"])

                # Get raw model output (probabilities)
                raw_output = model(x)
                probs = raw_output.cpu().numpy().flatten()  # Ensure 1D array
                preds = (probs > 0.5).astype(int)

                val_probs.append(probs)
                val_preds_comprehensive.append(preds)
                val_ys_comprehensive.append(
                    y.cpu().numpy().astype(int).flatten()
                )  # Ensure 1D array

                if i == 1000:  # Same limit as validation loop
                    break

        # Concatenate all predictions
        val_probs = np.concatenate(val_probs)
        val_preds_comprehensive = np.concatenate(val_preds_comprehensive)
        val_ys_comprehensive = np.concatenate(val_ys_comprehensive)

        return val_probs, val_preds_comprehensive, val_ys_comprehensive

    def _save_model_metadata(
        self, epoch: int, current_val_loss: float, report_path: str
    ):
        """
        Save metadata about the best model to a JSON file.

        Args:
            epoch: Current training epoch
            current_val_loss: Current validation loss
            report_path: Path to the comprehensive metrics report
        """
        metadata = {
            "best_epoch": int(epoch),
            "best_val_loss": float(current_val_loss),
            "model_path": self.best_model_path,
            "metrics_report_path": report_path,
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
        }

        metadata_path = (
            f"{self.package_path}/results/{self.run_id}/scores/best_model_metadata.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Best model metadata saved to: {metadata_path}")

    def get_best_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current best model.

        Returns:
            Dictionary containing best model information
        """
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_model_path": self.best_model_path,
        }

    def generate_training_plots(
        self, filename: str = "training_curves.png"
    ) -> Optional[str]:
        """
        Generate training curve plots from the saved CSV data.

        Creates a flexible grid plot showing training/validation curves for all
        metrics saved in scores_essential.csv. Each metric gets its own panel
        with epochs on x-axis and both train/val versions plotted together.

        Args:
            filename: Name for the output plot file

        Returns:
            Path to the saved plot file, or None if plot generation failed
        """
        csv_path = (
            f"{self.package_path}/results/{self.run_id}/scores/scores_essential.csv"
        )
        plots_dir = f"{self.package_path}/results/{self.run_id}/plots"

        return plot_training_curves_from_csv(csv_path, plots_dir, filename)
