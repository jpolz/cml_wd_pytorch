import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(loss_dict, run_id, package_path):
    """
    Plot training history from loss dictionary and save the figure.

    Args:
        loss_dict (dict): Dictionary containing training history with keys like 'epoch', 'train_bce', 'val_bce', etc.
        run_id (str): Unique identifier for the training run.
        package_path (str): Base path of the package to construct the full save path.
    """

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
    save_path = str(package_path) + "/results/%s/plots/loss_curves.png" % run_id
    plt.savefig(save_path)
    plt.close()

    print("Training history plot saved to: ", save_path)

    return None