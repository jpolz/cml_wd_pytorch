import xarray as xr
import numpy as np
import glob

if __name__ == "__main__":
    data_dir = "/bg/fast/aihydromet/cml_wet_dry_radklim"
    target_file = "cml_radklim.zarr"

    # Get all netCDF files in the target directory
    files = glob.glob(f"{data_dir}/train_data_v2*.nc")
    ds = xr.open_dataset(files[0])  # Check if the first file can be opened
    print(ds)
    files.sort()  # Sort files to ensure consistent order
    ds = xr.concat([xr.open_dataset(f) for f in files], dim="sample_number")
    print(ds)
    ds.to_zarr(f"{data_dir}/{target_file}", mode="w", consolidated=True)
    print(f"Data saved to {data_dir}/{target_file}")