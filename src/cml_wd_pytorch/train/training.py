import os
import uuid
from datetime import datetime
from pathlib import Path
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
    dataset = ZarrDataset(path, load=load, indices=indices)
    # balance the dataset
    ref = dataset.ds['wet_radar'].values
    print('dataset length: ', len(dataset))
    print('wet ratio: ', np.sum(ref) / len(ref) * 100)

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

    model = cnn()
    # summary(model, input_size=(1, 2, 180))  # Example input size (batch_size, channels, sequence_length)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], amsgrad=True)

    dataloader_train = build_dataloader(
        config['data']['path_train'], 
        batch_size=config['training']['batch_size'] ,
        load=True, 
        random=True, 
        num_workers=config['data']['num_workers'], 
        indices=np.arange(8000000)
        )
    print('dataloader train length: ', len(dataloader_train))
    dataloader_val = build_dataloader(
        config['data']['path_val'], 
        batch_size=config['training']['batch_size'] ,
        load=True, 
        random=False, 
        num_workers=config['data']['num_workers'], 
        indices=np.arange(8000000, 9000000)
        )    
    print('dataloader val length: ', len(dataloader_val))

    loss_dict = {}
    loss_dict['train_bce'] = []
    loss_dict['val_bce'] = []
    loss_dict['train_acc'] = []
    loss_dict['val_acc'] = []
    loss_dict['train_tpr'] = []
    loss_dict['val_tpr'] = []
    loss_dict['train_tnr'] = []
    loss_dict['val_tnr'] = []

    for epoch in range(config['training']['epochs']):
        losses = []
        preds = []
        ys = []
        for i, batch in tqdm(enumerate(dataloader_train)):
            x = batch[0].to(device).squeeze() # cml input
            y = batch[1].to(device).squeeze() # reference labels
            loss, pred = cnn.train_step(model, x, y, optimizer)
            losses.append(loss)
            preds.append(pred)
            ys.append(y.cpu().numpy())

        loss_dict['train_bce'].append(sum(losses) / len(losses))
        acc, tpr, tnr = acc(preds, ys)
        loss_dict['train_acc'].append(acc)
        loss_dict['train_tpr'].append(tpr)
        loss_dict['train_tnr'].append(tnr)

        test_losses = []
        test_preds = []
        test_ys = []
        for i, batch in tqdm(enumerate(dataloader_train)):
            x = batch[0].to(device).squeeze()
            y = batch[1].to(device).squeeze()
            loss, pred = cnn.test_step(model, x, y)
            test_losses.append(loss)
            test_preds.append(pred)
            test_ys.append(y.cpu().numpy())
            if i == 1000:
                break

        
        loss_dict[val_bce].append(sum(test_losses) / len(test_losses))
        acc, tpr, tnr = acc(test_preds, test_ys)
        loss_dict['val_acc'].append(acc)
        loss_dict['val_tpr'].append(tpr)
        loss_dict['val_tnr'].append(tnr)

        print(f'Test scores after Epoch {epoch}: {loss_dict["val_bce"][-1]:.4f}, '
              f'Acc: {loss_dict["val_acc"][-1]:.4f}, '
              f'TPR: {loss_dict["val_tpr"][-1]:.4f}, '
              f'TNR: {loss_dict["val_tnr"][-1]:.4f}'
              )
        print(f'Train scores after Epoch {epoch}: {loss_dict["train_bce"][-1]:.4f}, '
              f'Acc: {loss_dict["train_acc"][-1]:.4f}, '
              f'TPR: {loss_dict["train_tpr"][-1]:.4f}, '
              f'TNR: {loss_dict["train_tnr"][-1]:.4f}'
              )

        

            # forward pass
