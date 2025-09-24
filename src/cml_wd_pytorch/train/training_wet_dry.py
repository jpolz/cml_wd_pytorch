import argparse
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from cml_wd_pytorch.evaluation.scores import MetricsCalculator
from cml_wd_pytorch.models.cnn import cnn
from cml_wd_pytorch.train.model_logger import BestModelLogger
from cml_wd_pytorch.train.train_loader import build_dataloader_wd as build_dataloader
from cml_wd_pytorch.train.train_utils import EarlyStopping, MetricTracker


def process_batch(batch, device, reflength, threshold=0.1):
    """
    Process a batch to extract input features and target labels.

    Args:
        batch: Batch from dataloader containing [cml_input, _, rain_rate, cml_rain_rate]
        device: Device to move tensors to
        reflength: Number of recent time steps to consider for rain rate calculation
        threshold: Threshold in mm/h for wet/dry classification (default: 0.1)

    Returns:
        tuple: (x, y) where x is CML input and y is wet/dry labels
    """
    x = batch[0].to(device).squeeze()  # CML input features

    # Calculate rain rate: mean over last reflength values * 60 (mm/h conversion)
    y = (
        batch[2].to(device).squeeze()[:, -reflength:].mean(dim=-1) * 60
    ) > threshold  # Convert to wet/dry labels using threshold

    return x, y


if __name__ == "__main__":
    ##########################
    # Parse command line args #
    ##########################
    parser = argparse.ArgumentParser(description="Train wet/dry classification model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: src/cml_wd_pytorch/config/config.yml)",
    )
    args = parser.parse_args()

    ##########################
    # Set up experiment run  #
    ##########################
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("device: ", device)
    # get date string
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("date: ", date_str)
    # generate run id
    run_id = date_str + str(uuid.uuid4())
    print("run id: ", run_id)

    package_path = Path(
        os.path.abspath(__file__)
    ).parent.parent.parent.parent.absolute()

    # load config yml
    if args.config is not None:
        config_path = args.config
    else:
        config_path = str(package_path) + "/src/cml_wd_pytorch/config/config.yml"

    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if (
        not os.path.exists(str(package_path) + "/results/%s/" % run_id)
        and not config["experiment"]["debug"]
    ):
        os.makedirs(str(package_path) + "/results/%s/plots" % run_id)
        os.makedirs(str(package_path) + "/results/%s/models" % run_id)
        os.makedirs(str(package_path) + "/results/%s/scores" % run_id)
        # code to copy config.yml to results folder
        with open(str(package_path) + "/results/%s/config.yml" % run_id, "w") as f:
            config["experiment"]["run_id"] = run_id
            yaml.dump(config, f)

    #######################
    # dataloader and model
    #######################

    model = cnn(
        final_act="sigmoid",
    )
    # summary(model, input_size=(1, 2, 180))  # Example input size (batch_size, channels, sequence_length)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"], amsgrad=True
    )
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping"]["patience"],
        min_delta=config["training"]["early_stopping"]["min_delta"],
        mode=config["training"]["early_stopping"]["mode"],
    )

    dataloader_train = build_dataloader(
        config["data"]["path_train"],
        batch_size=config["training"]["batch_size"],
        load=True,
        random=True,
        num_workers=config["data"]["num_workers"],
        indices=np.arange(1000000),
        reflength=config["data"]["reflength"],
    )
    print("dataloader train length: ", len(dataloader_train))
    dataloader_val = build_dataloader(
        config["data"]["path_val"],
        batch_size=config["training"]["batch_size"],
        load=True,
        random=True,
        num_workers=config["data"]["num_workers"],
        indices=np.arange(1000000),
        reflength=config["data"]["reflength"],
    )
    print("dataloader val length: ", len(dataloader_val))

    metrics = MetricTracker(metrics=config["training"]["metrics"]["tracked_metrics"])

    # Initialize comprehensive metrics calculator for best epoch reporting
    comprehensive_metrics = MetricsCalculator(
        metrics=(
            MetricsCalculator.BASIC_METRICS
            + MetricsCalculator.ADVANCED_METRICS
            + MetricsCalculator.METEOROLOGICAL_METRICS
        )
    )

    # Initialize best model logger
    best_model_logger = BestModelLogger(
        package_path=str(package_path),
        run_id=run_id,
        comprehensive_metrics=comprehensive_metrics,
    )

    ########################
    # start training loop  #
    ########################

    for epoch in range(config["training"]["epochs"]):
        metrics.reset()

        # Training loop
        model.train()
        for i, batch in tqdm(enumerate(dataloader_train)):
            x, y = process_batch(batch, device, config["data"]["reflength"])
            loss, pred = cnn.train_step(model, x, y, optimizer)
            metrics.append(loss, "bce", "train")

        # Validation loop
        model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader_val)):
                x, y = process_batch(batch, device, config["data"]["reflength"])
                loss, pred = cnn.test_step(model, x, y)
                metrics.append(loss, "bce", "val")

                if i == 1000:
                    break

        # Log only BCE loss metrics
        metrics.log_epoch(
            epoch, {}, {}
        )  # Empty dicts since we only track BCE internally
        current_metrics = metrics.get_metrics()

        # Print simplified progress
        train_loss = current_metrics["train_bce"][-1]
        val_loss = current_metrics["val_bce"][-1]
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if not config["experiment"]["debug"]:
            # Check if this is the best model so far and handle all metrics logging
            current_val_loss = val_loss
            best_model_logger.check_and_update_best_model(
                model=model,
                dataloader_val=dataloader_val,
                process_batch_fn=process_batch,
                device=device,
                config=config,
                epoch=epoch,
                current_val_loss=current_val_loss,
                train_bce=train_loss,
            )

        # early stopping check
        monitor_metric = config["training"]["early_stopping"]["monitor"]
        if early_stopping(metrics.get_latest(monitor_metric, "val"), epoch):
            print("Early stopping triggered at epoch:", epoch)
            print(f"Best val_{monitor_metric}:", early_stopping.best_score)
            print("Best epoch:", early_stopping.best_epoch)
            break

    # Print final summary about the best model
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    best_model_info = best_model_logger.get_best_model_info()
    if best_model_info["best_epoch"] >= 0:
        print(f"Best model found at epoch: {best_model_info['best_epoch']}")
        print(f"Best validation loss: {best_model_info['best_val_loss']:.6f}")
        print(f"Best model saved to: {best_model_info['best_model_path']}")
        print(
            f"Comprehensive metrics report available at: {str(package_path)}/results/{run_id}/scores/best_epoch_comprehensive_metrics.txt"
        )
    else:
        print("No model was saved (debug mode or no improvement)")
    print("=" * 80)
