import xarray as xr
import numpy as np

if __name__ == "__main__":
    # Create a dummy dataset with random data
    n_samples = 1000
    n_channels = 2
    n_timesteps = 180

    # Create random data
    data = np.random.rand(n_samples, n_channels, n_timesteps)

    # Create an xarray Dataset
    ds = xr.Dataset(
        {
            "tl": (("sample", "channel", "time"), data)
        },
        coords={
            "sample": np.arange(n_samples),
            "channel": np.arange(n_channels),
            "time": np.arange(n_timesteps)
        }
    )

    # Save the dataset to a Zarr file
    ds.to_zarr('dummy_data.zarr', mode='w', consolidated=True)  # Use consolidated=True for better performance