import os
import uuid
from datetime import datetime
from pathlib import Path
import yaml
from torchinfo import summary
import torch
from torch.utils.data import DataLoader

from cml_wd_pytorch.dataloader.dataloaderzarr import ZarrDataset
from cml_wd_pytorch.models.cnn import cnn



def build_dataloader(path, batch_size=100, load=True, random=False, num_workers=40):
    """
    Build a dataloader for the given path and number of CML channels.
    Args:
        path (str): Path to the dataset.
        n_cml (int): Number of CML channels.
        batch_size (int): Batch size for the dataloader.
        img_size (int): Size of the images (default is 11).
        load (bool): Whether to load the dataset or not.
        random (bool): Whether to use a random sampler or not.
        num_workers (int): Number of workers for the dataloader.
    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = ZarrDataset(path, load=load,)
    print(len(dataset))
    if random:
        sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=100*batch_size)
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
    summary(model, input_size=(1, 2, 180))  # Example input size (batch_size, channels, sequence_length)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], amsgrad=True)

    dataloader_train = build_dataloader(config['data']['path_train'], batch_size=config['training']['batch_size'] ,load=True, random=False, num_workers=config['data']['num_workers'])
    print('dataloader train length: ', len(dataloader_train))
    dataloader_val = build_dataloader(config['data']['path_val'], batch_size=config['training']['batch_size'] ,load=True, random=False, num_workers=config['data']['num_workers'])
    print('dataloader val length: ', len(dataloader_val))

    loss_dict = {}
    for epoch in range(config['training']['epochs']):
        losses = []
        for i, batch in enumerate(dataloader_train):
            x = batch[0].to(device).squeeze() # cml input
            y = batch[1].to(device).squeeze() # reference labels
            loss, pred = cnn.train_step(model, x, y, optimizer)
            losses.append(loss)
        loss_dict[f'epoch_{epoch}'] = sum(losses) / len(losses)

        test_losses = []
        for i, batch in enumerate(dataloader_val):
            x = batch[0].to(device).squeeze()
            y = batch[1].to(device).squeeze()
            loss, pred = cnn.test_step(model, x, y)
            test_losses.append(loss)
        loss_dict[f'test_{epoch}'] = sum(test_losses) / len(test_losses)

        print(f'Test Loss after Epoch {epoch}: {loss_dict[f"test_{epoch}"]:.4f}')
        print(f'Loss after Epoch {epoch}: {loss_dict[f"epoch_{epoch}"]:.4f}')

        

            # forward pass
