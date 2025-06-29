import os
import uuid
from datetime import datetime
from pathlib import Path
import pandas as pd
import yaml
from torchinfo import summary
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from cml_wd_pytorch.dataloader.dataloaderzarr import ZarrDataset
from cml_wd_pytorch.models.cnn import cnn

def acc(preds, ys):
    """
    Calculate accuracy, true positive rate (TPR), and true negative rate (TNR) for the predictions.
    Args:
        preds (list): List of predicted labels.
        ys (list): List of true labels.
    Returns:
        tuple: Accuracy, TPR, and TNR.
    """
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    acc = np.mean(preds == ys)
    tpr = np.sum((preds == 1) & (ys == 1)) / np.sum(ys == 1) if np.sum(ys == 1) > 0 else 0
    tnr = np.sum((preds == 0) & (ys == 0)) / np.sum(ys == 0) if np.sum(ys == 0) > 0 else 0
    return acc, tpr, tnr


def build_dataloader(path, batch_size=100, load=True, random=False, num_workers=40, indices=None):
    """
    Build a dataloader for the given path and number of CML channels.
    Args:
        path (str): Path to the dataset.
        batch_size (int): Batch size for the dataloader.
        load (bool): Whether to load the dataset or not.
        random (bool): Whether to use a random sampler or not.
        num_workers (int): Number of workers for the dataloader.
    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = ZarrDataset(path, load=False,)
    # balance the dataset
    ref = dataset.ds['wet_radar'].values
    print('dataset length: ', len(dataset))
    print('wet ratio: ', np.sum(ref) / len(ref) * 100)

    rs = dataset.ds['radar'].values[:,-1]
    print('radar rain rate mean: ', np.mean(rs))
    print('radar rain rate nan ratio: ', np.sum(np.isnan(rs)) / len(rs) * 100)

    # get indices without nan values in rs
    indices2 = np.where(~np.isnan(rs))[0]
    print('indices2 length: ', len(indices2))

    # intersect indices with indices2
    if indices is not None:
        indices = np.intersect1d(indices, indices2)
    else:
        indices = indices2

    dataset = ZarrDataset(path, load=load, indices=indices)
    # balance the dataset
    ref = dataset.ds['wet_radar'].values
    print('dataset length: ', len(dataset))
    print('wet ratio: ', np.sum(ref) / len(ref) * 100)

    rs = dataset.ds['radar'].values[:,-1]
    print('radar rain rate mean: ', np.mean(rs))
    print('radar rain rate nan ratio: ', np.sum(np.isnan(rs)) / len(rs) * 100)

    print(len(dataset))
    if random:
        sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=1000*batch_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)  #each worker loads n-batch images
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":

    ####################
    # Set up experiment run
    ####################
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('device: ', device)
    # get date string
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print('date: ', date_str)
    # generate run id
    run_id = date_str+str(uuid.uuid4())
    print('run id: ', run_id)

    package_path = Path(os.path.abspath(__file__)).parent.parent.parent.parent.absolute()

    # load config yml
    with open(str(package_path)+'/src/cml_wd_pytorch/config/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    if not os.path.exists(str(package_path)+'/results/%s/' % run_id) and not config['experiment']['debug']:
        os.makedirs(str(package_path)+'/results/%s/plots' % run_id)
        os.makedirs(str(package_path)+'/results/%s/models' % run_id)
        os.makedirs(str(package_path)+'/results/%s/scores' % run_id)
        # code to copy config.yml to results folder
        with open(str(package_path)+'/results/%s/config.yml' % run_id, 'w') as f:
            config['experiment']['run_id'] = run_id
            yaml.dump(config, f)

    #######################
    # dataloader and model
    #######################

    model = cnn(final_act='relu')
    # summary(model, input_size=(1, 2, 180))  # Example input size (batch_size, channels, sequence_length)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], amsgrad=True)

    df = pd.read_csv('/bg/fast/aihydromet/cml_wet_dry_radklim/cml_radklim_indices.csv')
    ind = df['indices'].values

    print('indices length: ', len(ind))
    train_indices = np.intersect1d(np.arange(8000000), ind)
    print('train indices length: ', len(train_indices))
    val_indices = np.intersect1d(np.arange(8000000, 9000000), ind)
    print('val indices length: ', len(val_indices))

    dataloader_train = build_dataloader(
        config['data']['path_train'], 
        batch_size=config['training']['batch_size'] ,
        load=True, 
        random=True, 
        num_workers=config['data']['num_workers'], 
        indices=train_indices
        )
    print('dataloader train length: ', len(dataloader_train))
    dataloader_val = build_dataloader(
        config['data']['path_val'], 
        batch_size=config['training']['batch_size'] ,
        load=True, 
        random=False, 
        num_workers=config['data']['num_workers'], 
        indices=val_indices
        )    
    print('dataloader val length: ', len(dataloader_val))

    loss_dict = {}
    loss_dict['train_mse'] = []
    loss_dict['val_mse'] = []
    loss_dict['train_pred_mean'] = []
    loss_dict['val_pred_mean'] = []


    for epoch in range(config['training']['epochs']):
        losses = []
        preds = []
        ys = []
        for i, batch in tqdm(enumerate(dataloader_train)):
            x = batch[0].to(device).squeeze() # cml input
            y = batch[1].to(device).squeeze() # reference labels
            r = batch[2].to(device).squeeze()*60 # radar rain rate
            loss, pred = cnn.train_step(model, x, r, optimizer, loss='mse')
            losses.append(loss)
            preds.append(pred.detach().cpu().numpy())
            ys.append(r.cpu().numpy())

        loss_dict['train_mse'].append(sum(losses) / len(losses))
        loss_dict['train_pred_mean'].append(np.mean(np.concatenate(preds)))

        test_losses = []
        test_preds = []
        test_ys = []
        for i, batch in tqdm(enumerate(dataloader_val)):
            x = batch[0].to(device).squeeze()
            y = batch[1].to(device).squeeze()
            r = batch[2].to(device).squeeze()*60 # radar rain rate
            loss, pred = cnn.test_step(model, x, r, loss='mse')
            test_losses.append(loss)
            test_preds.append(pred.cpu().numpy())
            test_ys.append(r.cpu().numpy())
            if i == 1000:
                break

        
        loss_dict['val_mse'].append(sum(test_losses) / len(test_losses))
        loss_dict['val_pred_mean'].append(np.mean(np.concatenate(test_preds)))


        print(f'Test scores after Epoch {epoch}: {loss_dict["val_mse"][-1]:.4f}, '
              f'Pred Mean: {loss_dict["val_pred_mean"][-1]:.4f}, ')
        print(f'Train scores after Epoch {epoch}: {loss_dict["train_mse"][-1]:.4f}, '
              f'Pred Mean: {loss_dict["train_pred_mean"][-1]:.4f}, ')

