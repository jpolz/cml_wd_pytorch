"""
create_dataset.py
-----------------
This script processes commercial microwave link (CML) and weather radar data to generate training datasets for machine learning applications in rainfall detection and estimation. It loads, synchronizes, and preprocesses CML and radar data, applies rolling window extraction, balances wet/dry classes, and saves the resulting dataset as NetCDF files for each month and year.

Functions:
    _rolling_window(a, window):
        Efficiently creates a rolling window view of the last axis of array 'a' with the specified window size.
    balance_classes(a, boo):
        Balances the dataset along the 'sample_number' dimension so that the number of positive and negative samples (as defined by boolean array 'boo') is equal.

Main Workflow:
    - Iterates over years and months to process data in batches.
    - Loads and synchronizes CML and radar datasets.
    - Computes rolling medians and normalizes CML signal levels.
    - Extracts rolling windows of time series data for model input.
    - Balances the dataset for wet/dry radar events.
    - Saves the processed dataset to NetCDF files for downstream ML tasks.

Dependencies:
    - xarray
    - einops
    - numpy
    - tqdm
    - scikit-learn
"""
import xarray as xr
import einops
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle

def _rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(
        a, shape=shape, strides=strides, writeable=False
    )

def balance_classes(a, boo):
    lsn=len(a.sample_number)
    ind = np.arange(lsn)
    ind_true = shuffle(ind[boo])
    ind_false = ind[~boo]
    ind_true = ind_true[:np.sum(~boo)]
    print(1-(2*len(ind_false)/lsn))
    return a.isel(sample_number=np.concatenate([ind_true,ind_false]))

last_hour = range(-1,-60,-1)

if __name__ == "__main__":
    for year in ['2018','2019','2020']:
        for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
            cml_batch = shuffle(np.arange(3800))[:100]
            print('cml indices length: ', len(cml_batch))
            ds_rad = xr.open_dataset('/pd/data/CML/data/reference/radklim_yw_1min/2018_2020_radklim_yw_1_min.nc', engine='netcdf4')
            ds_cml1 = xr.open_dataset('/pd/data/CML/data/processed/proc2021.001/proc_hess_amt/proc_cnn_gapstandard_'+year+'_'+month+'.nc', engine='netcdf4')
            ds_rad = ds_rad.reindex(time=ds_cml1.time.values,method='nearest',tolerance='10s')
            ds_cml1['radar_rainfall'] = ds_rad.rainfall_amount
            print('loading cml data for year', year, 'month', month)
            ds_cml1 = ds_cml1.load()

            print('cml data loaded')
            ds_cml = ds_cml1.isel(cml_id=cml_batch).copy()

            ds_cml['r_median_tl'] = ds_cml.txrx.rolling(time=60*72, min_periods=60*3).median()
            ds_cml['tl_norm'] = ds_cml['txrx']-ds_cml['r_median_tl']
            print(ds_cml)
            # ds_cml = ds_cml.sel(time=slice('2020-05-20','2020-05-30'))
            tl = einops.rearrange(ds_cml.tl_norm.values, 'c i t -> t i c')
            r = einops.rearrange(ds_cml.rainfall_amount.values, 'c i t -> t i c')
            rr = ds_cml.radar_rainfall.values

            l = []
            rs = []
            rrs = []
            times = []
            print('rolling window')
            for i in tqdm(range(tl.shape[1])):
                tll = []
                rll = []
                for j in range(tl.shape[2]):
                    tll.append(_rolling_window(tl[:,i,j], 180))
                    rll.append(_rolling_window(r[:,i,j], 180))
                rrs.append(_rolling_window(rr[:,i], 180))
                times.append(_rolling_window(ds_cml.time.values, 180)[-1])
                l.append(tll)
                rs.append(rll)
            l = einops.rearrange(np.array(l), 'i c b t -> (b i) t c')
            rs = einops.rearrange(np.array(rs), 'i c b t -> (b i) t c')
            rrs = einops.rearrange(np.array(rrs), 'i b t -> (b i) t')
            times = einops.rearrange(np.array(times), 'i b -> (b i)')
            print('times shape:', times.shape)
            ds = xr.Dataset()
            ds['tl'] = ('sample_number','timestep','channel_id',), l
            ds['radar'] = ('sample_number','timestep',), rrs
            ds['cml_rain'] = ('sample_number','timestep','channel_id',), rs
            ds['tl_valid'] = ds.tl.isnull().sum(dim='timestep').sum(dim='channel_id')==0
            # ds['sample_time'] = ('sample_number',), times
            ds = ds.sel(sample_number=ds.sample_number[ds.tl_valid])
            ds['wet_radar'] = ds.radar.isel(timestep=last_hour).sum('timestep')>0.1

            ds = balance_classes(ds, ~ds.wet_radar.values)
            ds['tl'] = ds.tl.transpose('sample_number', 'channel_id', 'timestep')
            print('saving dataset for year', year, 'month', month)
            ds.to_netcdf('/bg/fast/aihydromet/cml_wet_dry_radklim/train_data_v3_'+year+'_'+month+'.nc')
            print('dataset saved for year', year, 'month', month)