import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

device = torch.device("cpu")


class MultiZarrDataset(Dataset):
    """
    PyTorch Dataset for loading multiple Zarr datasets.

    Args:
        paths (list): List of paths to Zarr datasets.
        indices (array-like, optional): Sample indices to load. Default: None (all samples).
        load (bool): Whether to load data into memory. Default: False.
    """

    def __init__(self, paths, indices=None, load=False):
        self.zarr_paths = paths
        if indices is None:
            self.ds = xr.concat(
                [xr.open_zarr(path) for path in self.zarr_paths], dim="sample"
            )
        else:
            self.ds = xr.concat(
                [xr.open_zarr(path) for path in self.zarr_paths], dim="sample"
            ).isel(sample=indices)
        if load:
            self.ds.load()
        self.n_samples = len(self.ds.sample)

    def __len__(self):
        """Return number of samples in dataset."""
        length = self.n_samples
        return length

    def __getitem__(self, index):
        """Load and return a single sample."""
        sample = self.ds.isel(sample=[index])
        sample = torch.tensor(sample["tl"].values, dtype=torch.float32)
        # TODO: complete this to return all necessary data

        return sample

    def close(self):
        """Close the xarray dataset when done."""
        self.ds.close()


class ZarrDataset(Dataset):
    """
    PyTorch Dataset for loading single Zarr dataset with CML and radar data.

    Args:
        path (str): Path to Zarr dataset.
        indices (array-like, optional): Sample indices to load. Default: None (all samples).
        load (bool): Whether to load data into memory. Default: False.
    """

    def __init__(self, path, indices=None, load=False):
        self.zarr_paths = path
        if indices is None:
            self.ds = xr.open_zarr(path)
        else:
            self.ds = xr.open_zarr(path).isel(sample_number=indices)
        if load:
            self.ds.load()
        self.n_samples = len(self.ds.sample_number)

    def __len__(self):
        """Return number of samples in dataset."""
        length = self.n_samples
        return length

    def __getitem__(self, index):
        """
        Load and return CML, reference, radar, and CML rain data for one sample.
        Args:
            index (int): Sample index.
        Returns:
            tuple: (cml, ref, r, r_cml) tensors.
        """
        sample = self.ds.isel(sample_number=[index])
        cml = torch.tensor(sample["tl"].values, dtype=torch.float32)
        ref = torch.tensor(sample["wet_radar"].values, dtype=torch.float32)
        r = torch.tensor(sample["radar"].values, dtype=torch.float32)
        r_cml = torch.tensor(
            np.nan_to_num(sample["cml_rain"].values), dtype=torch.float32
        )

        return cml, ref, r, r_cml

    def close(self):
        """Close the xarray dataset when done."""
        self.ds.close()
