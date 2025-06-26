import xarray as xr
import numpy as np

if __name__ == "__main__":
    # Create a dummy dataset with random data
    n_samples = 1000
    n_channels = 2
    n_timesteps = 180

    # Create random data
    data = np.random.rand(n_samples, n_channels, n_timesteps)
    # ref = np.random.randint(0, 2, size=n_samples)  # Binary reference labels (0 or 1)
    ref = np.ones(n_samples)  # All ones for simplicity, can be modified as needed

    # Create an xarray Dataset
    ds = xr.Dataset(
        {
            "tl": (("sample", "channel", "time"), data),
            "ref": (("sample",), ref)
        },
        coords={
            "sample": np.arange(n_samples),
            "channel": np.arange(n_channels),
            "time": np.arange(n_timesteps)
        }
    )

    # Save the dataset to a Zarr file
    ds.to_zarr('data/dummy_data.zarr', mode='w', consolidated=True)  # Use consolidated=True for better performance