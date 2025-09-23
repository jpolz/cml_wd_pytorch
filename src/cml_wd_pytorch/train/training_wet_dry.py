import argparse
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from cml_wd_pytorch.evaluation.scores import acc
from cml_wd_pytorch.models.cnn import cnn
from cml_wd_pytorch.train.train_loader import build_dataloader_wd as build_dataloader
from cml_wd_pytorch.train.train_utils import EarlyStopping, MetricTracker

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

    # TODO: Implement configurable metric computation pipeline
    # - Make thresholds configurable for wet/dry classification and prediction
    # - Support multiple metric functions (F1, precision, recall, AUC, etc.)
    # - Allow custom aggregation methods beyond mean (median, percentiles, etc.)
    # - Support different evaluation modes (per-timestep, per-cml, aggregated)

    for epoch in range(config["training"]["epochs"]):
        metrics.reset()
        preds = []
        ys = []
        wetcml = []
        for i, batch in tqdm(enumerate(dataloader_train)):
            x = batch[0].to(device).squeeze()  # cml input
            # TODO: Make rain rate calculation configurable (currently hard-coded formula)
            # Current: mean over last reflength values * 60 (mm/h conversion)
            y = (
                (
                    batch[2]
                    .to(device)
                    .squeeze()[:, -config["data"]["reflength"] :]
                    .mean(dim=-1)
                    * 60
                )
                > 0.1
            )  # TODO: Make wet/dry threshold configurable (currently hard-coded to 0.1 mm/h)
            wcml = (
                (
                    batch[3]
                    .to(device)
                    .squeeze()[:, -config["data"]["reflength"] :, 0]
                    .mean(dim=-1)
                    * 60
                )
                > 0.1
            )  # cml rain rate, not used in training  # TODO: Make CML threshold configurable
            loss, pred = cnn.train_step(model, x, y, optimizer)
            metrics.append(loss, "bce", "train")
            # TODO: Make prediction threshold configurable (currently hard-coded to 0.5)
            preds.append(pred.detach().cpu().numpy() > 0.5)
            ys.append(y.cpu().numpy() > 0.5)
            wetcml.append(wcml.cpu().numpy() > 0.5)

        # Calculate training metrics
        # TODO: Replace hard-coded metric calculation with configurable metric functions
        ac, tpr, tnr = acc(preds, ys)
        ac_wetcml, tpr_wetcml, tnr_wetcml = acc(wetcml, ys)

        # TODO: Make metric dictionary construction dynamic based on config["training"]["metrics"]["tracked_metrics"]
        # No need to pass train_metrics, let log_epoch compute from accumulated values
        train_metrics = {
            "acc": ac,
            "tpr": tpr,
            "tnr": tnr,
            "acc_cml": ac_wetcml,
            "tpr_cml": tpr_wetcml,
            "tnr_cml": tnr_wetcml,
        }

        test_preds = []
        test_ys = []
        wetcml = []
        for i, batch in tqdm(enumerate(dataloader_val)):
            x = batch[0].to(device).squeeze()
            # TODO: Make validation rain rate calculation configurable (should match training)
            # y = batch[1].to(device).squeeze() # reference labels
            y = (
                (
                    batch[2]
                    .to(device)
                    .squeeze()[:, -config["data"]["reflength"] :]
                    .mean(dim=-1)
                    * 60
                )
                > 0.1
            )  # TODO: Make validation wet/dry threshold configurable (should match training)
            wcml = (
                batch[3]
                .to(device)
                .squeeze()[:, -config["data"]["reflength"] :, 0]
                .mean(dim=-1)
                * 60
            ) > 0.1  # TODO: Make validation CML threshold configurable
            loss, pred = cnn.test_step(model, x, y)
            metrics.append(loss, "bce", "val")
            # TODO: Make validation prediction thresholds configurable (should match training)
            test_preds.append(pred.cpu().numpy() > 0.5)
            test_ys.append(y.cpu().numpy() > 0.5)
            wetcml.append(wcml.cpu().numpy() > 0.5)
            if i == 1000:
                break

        # Calculate validation metrics
        # TODO: Replace hard-coded validation metric calculation with configurable metric functions
        ac, tpr, tnr = acc(test_preds, test_ys)
        ac_wetcml, tpr_wetcml, tnr_wetcml = acc(wetcml, test_ys)

        # TODO: Make validation metric dictionary construction dynamic based on config
        val_metrics = {
            "acc": ac,
            "tpr": tpr,
            "tnr": tnr,
            "acc_cml": ac_wetcml,
            "tpr_cml": tpr_wetcml,
            "tnr_cml": tnr_wetcml,
        }  # Log epoch metrics - this will compute BCE from accumulated values
        metrics.log_epoch(
            epoch, train_metrics, val_metrics
        )  # Get current metrics for printing
        current_metrics = metrics.get_metrics()

        print(
            f"Test scores after Epoch {epoch}: {current_metrics['val_bce'][-1]:.4f}, "
            f"Acc: {current_metrics['val_acc'][-1]:.4f}, "
            f"TPR: {current_metrics['val_tpr'][-1]:.4f}, "
            f"TNR: {current_metrics['val_tnr'][-1]:.4f}"
        )
        print(
            f"Train scores after Epoch {epoch}: {current_metrics['train_bce'][-1]:.4f}, "
            f"Acc: {current_metrics['train_acc'][-1]:.4f}, "
            f"TPR: {current_metrics['train_tpr'][-1]:.4f}, "
            f"TNR: {current_metrics['train_tnr'][-1]:.4f}"
        )

        if not config["experiment"]["debug"]:
            # save model
            torch.save(
                model.state_dict(),
                str(package_path)
                + "/results/%s/models/model_epoch_%d.pth" % (run_id, epoch),
            )
            print(
                "Model saved to: ",
                str(package_path)
                + "/results/%s/models/model_epoch_%d.pth" % (run_id, epoch),
            )

            # save scores to csv
            df = pd.DataFrame(metrics.get_metrics())
            df.to_csv(
                str(package_path) + "/results/%s/scores/scores.csv" % run_id, index=True
            )
            print(
                "Scores saved to: ",
                str(package_path) + "/results/%s/scores/scores.csv" % run_id,
            )
            # plot training history
            from cml_wd_pytorch.train.plot_train import plot_training_history

            plot_training_history(
                metrics.get_metrics(),
                run_id,
                package_path,
            )

        # early stopping check
        monitor_metric = config["training"]["early_stopping"]["monitor"]
        if early_stopping(metrics.get_latest(monitor_metric, "val"), epoch):
            print("Early stopping triggered at epoch:", epoch)
            print(f"Best val_{monitor_metric}:", early_stopping.best_score)
            print("Best epoch:", early_stopping.best_epoch)
            break
