import os
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from cml_wd_pytorch.train.train_loader import build_dataloader_wd as build_dataloader
from cml_wd_pytorch.models.cnn import cnn
from cml_wd_pytorch.evaluation.scores import acc


if __name__ == "__main__":
    ####################
    # Set up experiment run
    ####################
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
    with open(str(package_path) + "/src/cml_wd_pytorch/config/config.yml", "r") as f:
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

    loss_dict = {}
    loss_dict["epoch"] = []
    loss_dict["train_bce"] = []
    loss_dict["val_bce"] = []
    loss_dict["train_acc"] = []
    loss_dict["val_acc"] = []
    loss_dict["train_tpr"] = []
    loss_dict["val_tpr"] = []
    loss_dict["train_tnr"] = []
    loss_dict["val_tnr"] = []
    # loss_dict['train_bce_cml'] = []
    # loss_dict['val_bce_cml'] = []
    loss_dict["train_acc_cml"] = []
    loss_dict["val_acc_cml"] = []
    loss_dict["train_tpr_cml"] = []
    loss_dict["val_tpr_cml"] = []
    loss_dict["train_tnr_cml"] = []
    loss_dict["val_tnr_cml"] = []

    for epoch in range(config["training"]["epochs"]):
        loss_dict["epoch"].append(epoch)
        losses = []
        preds = []
        ys = []
        wetcml = []
        for i, batch in tqdm(enumerate(dataloader_train)):
            x = batch[0].to(device).squeeze()  # cml input
            # y = batch[1].to(device).squeeze() # reference labels
            y = (
                batch[2]
                .to(device)
                .squeeze()[:, -config["data"]["reflength"] :]
                .mean(dim=-1)
                * 60
            ) > 0.1
            wcml = (
                batch[3]
                .to(device)
                .squeeze()[:, -config["data"]["reflength"] :, 0]
                .mean(dim=-1)
                * 60
            ) > 0.1  # cml rain rate, not used in training
            loss, pred = cnn.train_step(model, x, y, optimizer)
            losses.append(loss)
            preds.append(pred.detach().cpu().numpy() > 0.5)
            ys.append(y.cpu().numpy() > 0.5)
            wetcml.append(wcml.cpu().numpy() > 0.5)

        loss_dict["train_bce"].append(sum(losses) / len(losses))
        ac, tpr, tnr = acc(preds, ys)
        ac_wetcml, tpr_wetcml, tnr_wetcml = acc(wetcml, ys)
        loss_dict["train_acc"].append(ac)
        loss_dict["train_tpr"].append(tpr)
        loss_dict["train_tnr"].append(tnr)
        loss_dict["train_acc_cml"].append(ac_wetcml)
        loss_dict["train_tpr_cml"].append(tpr_wetcml)
        loss_dict["train_tnr_cml"].append(tnr_wetcml)

        test_losses = []
        test_preds = []
        test_ys = []
        wetcml = []
        for i, batch in tqdm(enumerate(dataloader_val)):
            x = batch[0].to(device).squeeze()
            # y = batch[1].to(device).squeeze() # reference labels
            y = (
                batch[2]
                .to(device)
                .squeeze()[:, -config["data"]["reflength"] :]
                .mean(dim=-1)
                * 60
            ) > 0.1
            wcml = (
                batch[3]
                .to(device)
                .squeeze()[:, -config["data"]["reflength"] :, 0]
                .mean(dim=-1)
                * 60
            ) > 0.1
            loss, pred = cnn.test_step(model, x, y)
            test_losses.append(loss)
            test_preds.append(pred.cpu().numpy() > 0.5)
            test_ys.append(y.cpu().numpy() > 0.5)
            wetcml.append(wcml.cpu().numpy() > 0.5)
            if i == 1000:
                break

        loss_dict["val_bce"].append(sum(test_losses) / len(test_losses))
        ac, tpr, tnr = acc(test_preds, test_ys)
        ac_wetcml, tpr_wetcml, tnr_wetcml = acc(wetcml, test_ys)
        loss_dict["val_acc_cml"].append(ac_wetcml)
        loss_dict["val_tpr_cml"].append(tpr_wetcml)
        loss_dict["val_tnr_cml"].append(tnr_wetcml)
        loss_dict["val_acc"].append(ac)
        loss_dict["val_tpr"].append(tpr)
        loss_dict["val_tnr"].append(tnr)

        print(
            f"Test scores after Epoch {epoch}: {loss_dict['val_bce'][-1]:.4f}, "
            f"Acc: {loss_dict['val_acc'][-1]:.4f}, "
            f"TPR: {loss_dict['val_tpr'][-1]:.4f}, "
            f"TNR: {loss_dict['val_tnr'][-1]:.4f}"
        )
        print(
            f"Train scores after Epoch {epoch}: {loss_dict['train_bce'][-1]:.4f}, "
            f"Acc: {loss_dict['train_acc'][-1]:.4f}, "
            f"TPR: {loss_dict['train_tpr'][-1]:.4f}, "
            f"TNR: {loss_dict['train_tnr'][-1]:.4f}"
        )

        # for key in loss_dict.keys():
        #     print(f'{key}: {loss_dict[key][-1]:.4f}')

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
            df = pd.DataFrame(loss_dict)
            df.to_csv(
                str(package_path) + "/results/%s/scores/scores.csv" % run_id, index=True
            )
            print(
                "Scores saved to: ",
                str(package_path) + "/results/%s/scores/scores.csv" % run_id,
            )

            # plot loss curves
            plt.figure(figsize=(10, 5))
            plt.plot(loss_dict["train_bce"], label="Train TNR", color="blue")
            plt.plot(loss_dict["val_bce"], label="Validation TNR", color="orange")
            # add RMSE
            plt.plot(
                loss_dict["train_tnr"],
                label="Train TNR",
                color="blue",
                linestyle="dotted",
            )
            plt.plot(
                loss_dict["val_tnr"],
                label="Validation TNR",
                color="orange",
                linestyle="dotted",
            )
            # add RMSE
            plt.plot(
                np.sqrt(loss_dict["train_tpr"]),
                label="Train TPR",
                color="blue",
                linestyle="--",
            )
            plt.plot(
                np.sqrt(loss_dict["val_tpr"]),
                label="Validation TPR",
                color="orange",
                linestyle="--",
            )
            # add pearson r
            plt.plot(loss_dict["train_acc"], label="Train ACC", color="green")
            plt.plot(loss_dict["val_acc"], label="Validation ACC", color="red")
            # add cml pearson r
            plt.plot(
                loss_dict["train_acc_cml"],
                label="Train CML acc",
                color="green",
                linestyle="--",
            )
            plt.plot(
                loss_dict["val_acc_cml"],
                label="Validation CML acc",
                color="red",
                linestyle="--",
            )
            plt.ylim(0, 1)
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.title("Loss Curves")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                str(package_path) + "/results/%s/plots/loss_curves.png" % (run_id,)
            )
            plt.close()
