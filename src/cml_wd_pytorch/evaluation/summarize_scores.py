import matplotlib.pyplot as plt
import pandas as pd
import yaml
import numpy as np

if __name__ == "__main__":
    run_ids = [
        "2025-07-01_11-34-26e4adbb69-33c0-441c-91d8-68555bbbc229",
        # "2025-07-01_16-06-11ebccdf8d-3a07-4ea5-a931-742c3e3cc7b0",
        "2025-07-02_10-47-42267dbc1d-1877-47d4-beba-924ffa3f4465",
        "2025-07-02_15-10-404ad8e353-cfb2-4173-9ba9-5dec9261a12e",
    ]

    model_names = {
        "2025-07-01_11-34-26e4adbb69-33c0-441c-91d8-68555bbbc229": "CNN, 64 FC width",
        # "2025-07-01_16-06-11ebccdf8d-3a07-4ea5-a931-742c3e3cc7b0": "CNN, 64 FC width LR 0.001",
        "2025-07-02_10-47-42267dbc1d-1877-47d4-beba-924ffa3f4465": "CNN, 128 FC width",
        "2025-07-02_15-10-404ad8e353-cfb2-4173-9ba9-5dec9261a12e": "CNN, 128 FC double CNN filters",
    }
    # load the configs
    configs = {}
    for run_id in run_ids:
        with open(f"results/{run_id}/config.yml", 'r') as f:
            configs[run_id] = yaml.safe_load(f)
    # load loss curves from yaml files
    loss_curves = {}
    # model_names = {}
    for run_id in run_ids:
        # load the scores from the csv file
        print(f"Loading scores for run {run_id}...")
        # model_names[run_id] = configs[run_id]['experiment']['name']
        df = pd.read_csv(f"results/{run_id}/scores/scores.csv")
        loss_curves[run_id+'_train_mse'] = df['train_mse'].tolist()
        loss_curves[run_id+'_validation_mse'] = df['val_mse'].tolist()
        loss_curves[run_id+'_train_pearson'] = df['train_pearson'].tolist()
        loss_curves[run_id+'_validation_pearson'] = df['val_pearson'].tolist()


    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    # plot the loss curves
    plt.figure(figsize=(10, 6))
    for i, run_id in enumerate(run_ids):
        plt.plot(loss_curves[run_id+'_validation_pearson'], label=model_names[run_id], color=colors[i], linestyle='--', linewidth=1)
        plt.plot(np.arange(3,len(loss_curves[run_id+'_validation_pearson'])-3), np.convolve(loss_curves[run_id+'_validation_pearson'], np.ones(5)/5, mode='full')[5:-5], color=colors[i], linestyle='-', linewidth=2)
    plt.xlabel('step')
    plt.ylabel('PCC')
    plt.yticks(np.arange(0.65, 0.75, 0.01))
    plt.ylim(0.65, 0.75)
    plt.legend()
    plt.grid()
    plt.savefig('results/summary/PCC_curves.png')


    # plot the loss curves
    plt.figure(figsize=(10, 6))
    for i, run_id in enumerate(run_ids):
        plt.plot(np.arange(3,len(loss_curves[run_id+'_validation_pearson'])-3), np.convolve(loss_curves[run_id+'_validation_pearson'], np.ones(25)/25, mode='full')[15:-15], label=model_names[run_id], color=colors[i], linestyle='--', linewidth=1)
        # plt.plot(np.arange(3,len(loss_curves[run_id+'_validation_pearson'])-3), np.convolve(loss_curves[run_id+'_validation_pearson'], np.ones(15)/15, mode='full')[10:-10], color=colors[i], linestyle='-', linewidth=2)
    plt.xlabel('step')
    plt.ylabel('PCC')
    plt.yticks(np.arange(0.65, 0.75, 0.01))
    plt.ylim(0.65, 0.75)
    plt.legend()
    plt.grid()
    plt.savefig('results/summary/PCC_curves_smooth.png')


    # plot the loss curves
    plt.figure(figsize=(10, 6))
    for i, run_id in enumerate(run_ids):
        plt.plot(loss_curves[run_id+'_validation_mse'], label=model_names[run_id], color=colors[i], linestyle='--', linewidth=1)
        plt.plot(np.arange(3,len(loss_curves[run_id+'_validation_mse'])-3), np.convolve(loss_curves[run_id+'_validation_mse'], np.ones(5)/5, mode='full')[5:-5], color=colors[i], linestyle='-', linewidth=2)
    plt.xlabel('step')
    plt.ylabel('MSE')
    plt.yticks(np.arange(0.4, 0.76, 0.02))
    plt.ylim(0.4, 0.76)
    plt.legend()
    plt.grid()
    plt.savefig('results/summary/MSE_curves.png')


    # plot the loss curves
    plt.figure(figsize=(10, 6))
    for i, run_id in enumerate(run_ids):
        plt.plot(np.arange(3,len(loss_curves[run_id+'_validation_mse'])-3), np.convolve(loss_curves[run_id+'_validation_mse'], np.ones(25)/25, mode='full')[15:-15], label=model_names[run_id], color=colors[i], linestyle='--', linewidth=1)
        # plt.plot(np.arange(3,len(loss_curves[run_id+'_validation_mse'])-3), np.convolve(loss_curves[run_id+'_validation_mse'], np.ones(5)/5, mode='full')[5:-5], color=colors[i], linestyle='-', linewidth=2)
    plt.xlabel('step')
    plt.ylabel('MSE')
    plt.yticks(np.arange(0.4, 0.76, 0.02))
    plt.ylim(0.4, 0.76)
    plt.legend()
    plt.grid()
    plt.savefig('results/summary/MSE_curves_smooth.png')

    # # plot the loss curves
    # plt.figure(figsize=(10, 6))
    # for i, run_id in enumerate(run_ids):
    #     plt.plot(loss_curves[run_id+'_mse'], label=model_names[run_id]+' ('+run_id+')', color=colors[i], linestyle='--', linewidth=1)
    #     plt.plot(np.arange(3,len(loss_curves[run_id+'_mse'])-3), np.convolve(loss_curves[run_id+'_mse'], np.ones(5)/5, mode='full')[5:-5], label=model_names[run_id]+' (smoothed)', color=colors[i], linestyle='-', linewidth=2)
    # plt.xlabel('step')
    # plt.ylabel('MSE Loss')
    # plt.title('MSE Loss Curves')
    # plt.legend()
    # plt.grid()
    # plt.savefig('results/summary/mse_loss_curves.png')

    # # plot the loss curves
    # plt.figure(figsize=(10, 6))
    # for i, run_id in enumerate(run_ids):
    #     plt.plot(loss_curves[run_id+'_loss'], label=model_names[run_id]+' ('+run_id+')', color=colors[i], linestyle='--', linewidth=1)
    #     plt.plot(np.arange(3,len(loss_curves[run_id+'_loss'])-3), np.convolve(loss_curves[run_id+'_loss'], np.ones(5)/5, mode='full')[5:-5], label=model_names[run_id]+' (smoothed)', color=colors[i], linestyle='-', linewidth=2)
    # plt.xlabel('step')
    # plt.ylabel('Loss')
    # plt.title('Loss Curves')
    # plt.legend()
    # plt.grid()
    # plt.savefig('results/summary/loss_curves.png')