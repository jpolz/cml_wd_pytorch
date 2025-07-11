import matplotlib.pyplot as plt
import pandas as pd
import yaml
import numpy as np

if __name__ == "__main__":
    run_ids = [
        "2025-07-04_11-26-26f2492292-eff6-413f-b2dc-d9ce98131f55",
        "2025-07-04_12-32-5046e093b9-e85e-4a53-9bfe-9c767798f5ad",
    ]

    model_names = {
        "2025-07-04_11-26-26f2492292-eff6-413f-b2dc-d9ce98131f55": "CNN, last 5min ref",
        "2025-07-04_12-32-5046e093b9-e85e-4a53-9bfe-9c767798f5ad": "CNN, last 60min ref",
    }
    # load the configs
    configs = {}
    for run_id in run_ids:
        with open(f"results/{run_id}/config.yml", "r") as f:
            configs[run_id] = yaml.safe_load(f)
    # load loss curves from yaml files
    loss_curves = {}
    scores = [
        "train_bce",
        "val_bce",
        "train_acc",
        "val_acc",
        "train_tpr",
        "val_tpr",
        "train_tnr",
        "val_tnr",
        "train_acc_cml",
        "val_acc_cml",
        "train_tpr_cml",
        "val_tpr_cml",
        "train_tnr_cml",
        "val_tnr_cml",
    ]
    scores_to_combine = {
        "train_tpr": "train_tpr_cml",
        "val_tpr": "val_tnr",
        "train_tnr": "train_tnr_cml",
        "val_tnr": "val_tnr_cml",
        "train_acc": "train_acc_cml",
        "val_acc": "val_acc_cml",
        "val_bce": "train_bce",
    }
    # model_names = {}
    for run_id in run_ids:
        # load the scores from the csv file
        print(f"Loading scores for run {run_id}...")
        # model_names[run_id] = configs[run_id]['experiment']['name']
        df = pd.read_csv(f"results/{run_id}/scores/scores.csv")
        for score in scores:
            if score in df.columns:
                loss_curves[run_id + "_" + score] = df[score].tolist()
            else:
                print(f"Warning: {score} not found in {run_id} scores.")

    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    for score in scores:
        if score in scores_to_combine:
            vmax = max([
                max([max(loss_curves[run_id + "_" + score]), max(loss_curves[run_id + "_" + scores_to_combine[score]])
                ])
                for run_id in run_ids
                if run_id + "_" + score in loss_curves
            ])
            vmin = min([
                min([min(loss_curves[run_id + "_" + score]), min(loss_curves[run_id + "_" + scores_to_combine[score]])
                ])
                for run_id in run_ids
                if run_id + "_" + score in loss_curves
            ])
        else:
            vmax = max([max(loss_curves[run_id + "_" + score]) for run_id in run_ids])
            vmin = min([min(loss_curves[run_id + "_" + score]) for run_id in run_ids])
        print(f"{score}: {vmin:.4f} - {vmax:.4f}")
        plt.figure(figsize=(10, 6))
        for i, run_id in enumerate(run_ids):
            plt.plot(
                loss_curves[run_id + "_" + score],
                label=model_names[run_id],
                color=colors[i],
                linestyle="--",
                linewidth=1,
            )
            plt.plot(
                np.arange(3, len(loss_curves[run_id + "_" + score]) - 3),
                np.convolve(
                    loss_curves[run_id + "_" + score], np.ones(5) / 5, mode="full"
                )[5:-5],
                color=colors[i],
                linestyle="-",
                linewidth=2,
            )
            if score in scores_to_combine:
                plt.plot(
                    np.arange(
                        3, len(loss_curves[run_id + "_" + scores_to_combine[score]]) - 3
                    ),
                    np.convolve(
                        loss_curves[run_id + "_" + scores_to_combine[score]],
                        np.ones(5) / 5,
                        mode="full",
                    )[5:-5],
                    color=colors[i],
                    linestyle=":",
                    linewidth=1,
                    label=scores_to_combine[score].replace("_", " ").title(),
                )
        plt.xlabel("step")
        plt.ylabel(score.replace("_", " ").title())
        plt.yticks(np.arange(vmin, vmax, (vmax - vmin) / 10))
        plt.ylim(vmin, vmax)
        plt.legend()
        plt.grid()
        if score in scores_to_combine:
            plt.savefig(f"results/summary/binary/{score}_curves_w_{scores_to_combine[score]}.png")
        else:
            plt.savefig(f"results/summary/binary/{score}_curves.png")
        plt.close()

        # plt.figure(figsize=(10, 6))
        # for i, run_id in enumerate(run_ids):
        #     plt.plot(
        #         np.arange(3, len(loss_curves[run_id + "_" + score]) - 3),
        #         np.convolve(
        #             loss_curves[run_id + "_" + score], np.ones(5) / 5, mode="full"
        #         )[5:-5],
        #         label=model_names[run_id],
        #         color=colors[i],
        #         linestyle="--",
        #         linewidth=1,
        #     )
        # plt.xlabel("step")
        # plt.ylabel(score.replace("_", " ").title())
        # plt.yticks(np.arange(vmin, vmax, (vmax - vmin) / 10))
        # plt.ylim(vmin, vmax)
        # plt.legend()
        # plt.grid()
        # plt.savefig(f"results/summary/binary/{score}_curves_smooth.png")
        # plt.close()
