import numpy as np
import torch
from torch.utils.data import DataLoader

from cml_wd_pytorch.dataset.zarrdataset import ZarrDataset


def build_dataloader(
    path,
    task_type="wd",
    batch_size=100,
    load=True,
    random=False,
    num_workers=40,
    indices=None,
    reflength=60,
):
    """
    Build a PyTorch DataLoader for CML training data.

    Args:
        path (str): Path to the Zarr dataset.
        task_type (str): Task type - 'wd' for wet/dry or 'rr' for rain rate. Default: 'wd'.
        batch_size (int): Batch size. Default: 100.
        load (bool): Whether to load dataset into memory. Default: True.
        random (bool): Whether to use random sampling. Default: False.
        num_workers (int): Number of workers for data loading. Default: 40.
        indices (array-like, optional): Sample indices to use. Default: None.
        reflength (int): Reference length for rainfall calculation. Default: 60.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    if task_type not in ["wd", "rr"]:
        raise ValueError("task_type must be 'wd' (wet/dry) or 'rr' (rain rate)")

    dataset = ZarrDataset(
        path,
        load=False,
    )
    # balance the dataset
    ref = dataset.ds["wet_radar"].values
    print("dataset length: ", len(dataset))
    print("wet ratio: ", np.sum(ref) / len(ref) * 100)

    rs = (
        dataset.ds["radar"].values[:, -reflength:].mean(axis=-1) * 60
    )  # sum over the last dimension to get the rainfall amount
    print("radar rain rate mean: ", np.mean(rs))
    print("radar rain rate nan ratio: ", np.sum(np.isnan(rs)) / len(rs) * 100)

    # get indices without nan values in rs
    indices2 = np.where(~np.isnan(rs))[0]
    print("indices2 length: ", len(indices2))

    # intersect indices with indices2
    if indices is not None:
        indices = np.intersect1d(indices, indices2)
    else:
        indices = indices2

    dataset = ZarrDataset(path, load=load, indices=indices)
    # balance the dataset
    ref = dataset.ds["wet_radar"].values
    print("dataset length: ", len(dataset))
    print("wet ratio: ", np.sum(ref) / len(ref) * 100)

    rs = (
        dataset.ds["radar"].values[:, -reflength:].mean(axis=-1) * 60
    )  # sum over the last dimension to get the rainfall amount
    print("radar rain rate mean: ", np.mean(rs))
    print("radar rain rate nan ratio: ", np.sum(np.isnan(rs)) / len(rs) * 100)

    print(len(dataset))
    if random:
        sampler = torch.utils.data.RandomSampler(
            dataset, replacement=False, num_samples=1000 * batch_size
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
        )  # each worker loads n-batch images
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    return dataloader


# Backward compatibility functions
def build_dataloader_rr(*args, **kwargs):
    """Backward compatibility wrapper for rain rate dataloader."""
    return build_dataloader(*args, task_type="rr", **kwargs)


def build_dataloader_wd(*args, **kwargs):
    """Backward compatibility wrapper for wet/dry dataloader."""
    return build_dataloader(*args, task_type="wd", **kwargs)
