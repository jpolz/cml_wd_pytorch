# CML Wet-Dry PyTorch

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)

A PyTorch re-implementation and improvement of commercial microwave link (CML) wet-dry detection based on [Polz et al. 2020](https://doi.org/10.5194/amt-13-3835-2020).

## 🔬 Overview

This project provides machine learning tools for rainfall detection and estimation using commercial microwave link (CML) data combined with weather radar observations. Commercial microwave links are telecommunication infrastructure that can be used as opportunistic sensors for precipitation monitoring, offering valuable insights for meteorological applications.

The package implements deep learning approaches to:
- **Detect precipitation events** (wet/dry classification)
- **Estimate rainfall rates** from CML signal attenuation
- **Process and analyze** large meteorological datasets efficiently

## ✨ Features

- 🧠 **Deep Learning Models**: Custom CNN architecture for CML time series analysis
- 📊 **Efficient Data Processing**: Zarr-based dataset handling for large meteorological data
- 🔧 **Configurable Training**: YAML-based experiment configuration
- 📈 **Comprehensive Evaluation**: Multiple metrics including accuracy, TPR, TNR, and correlation
- 🚀 **Production Ready**: Inference pipeline for integration in operational deployment

## 🛠️ Installation

### Requirements

- Python ≥ 3.12
- CUDA-capable GPU (recommended for training)

### Install from source

```bash
git clone https://github.com/jpolz/cml_wd_pytorch.git
cd cml_wd_pytorch
pip install -e .
```

### Dependencies

The project automatically installs:
- PyTorch
- XArray
- Zarr
- NumPy
- Matplotlib
- NetCDF4
- Einops
- scikit-learn
- TQDM
- PyYAML

## 🚀 Quick Start

### 1. Configuration

Edit the configuration file to match your data paths:

```yaml
# src/cml_wd_pytorch/config/config.yml
data:
  path_train: "/path/to/training/data.zarr"
  path_val: "/path/to/validation/data.zarr"
  reflength: 60

training:
  batch_size: 100
  epochs: 500
  learning_rate: 0.0001
```

### 2. [WIP] Training a Model

#### Wet/Dry Classification
```python
from cml_wd_pytorch.train.training_wet_dry import main

# Run training with configuration
main()
```

#### Rain Rate Estimation
```python
from cml_wd_pytorch.train.training_rain_rate import main

# Run training with configuration
main()
```

### 3. Running Inference

```python
from cml_wd_pytorch.inference.run_inference import cnn_wd
import xarray as xr

# Load your CML data
data = xr.open_dataset("your_cml_data.nc")

# Ensure data is in the expected format
# Example: data should be an xarray DataArray of total loss (TL) with dimensions [time, channel_id, cml_id]
data = data["tl"].transpose("time", "channel_id", "cml_id")

# Run inference using either a model path, run_id, or URL:

# Option 1: Provide the path to a trained model (.pth)
results = cnn_wd("path/to/trained/model.pth", data)

# Option 2: Provide a run_id (will automatically locate model and config in results/{run_id}/)
results = cnn_wd("2025-08-06_11-03-498c8c7046-872a-464e-b3b6-d6eeaff6a23b", data)

# Option 3: Provide a URL to download and cache the model
results = cnn_wd("https://github.com/user/repo/releases/download/v1.0/model.pth", data)

# Optional parameters:
# - config_path: Custom config file path
# - force_download: Force re-download of cached models
# results = cnn_wd("https://example.com/model.pth", data, force_download=True)
```

#### Model Caching

When using URLs, models are automatically cached in `~/.cml_wd_pytorch/models/` to avoid repeated downloads:

```python
from cml_wd_pytorch.inference.run_inference import list_cached_models, clear_model_cache

# List cached models
cached_models = list_cached_models()
print(f"Cached models: {len(cached_models)}")

# Clear cache if needed
clear_model_cache()
```

## 📊 Data Format for training

The package expects data in Zarr format with the following structure:

```
dataset.zarr/
├── sample_number/     # Sample dimension
├── channel_id/        # CML channel dimension  
├── timestep/          # Time dimension
├── tl/               # CML signal attenuation [sample_number, channel_id, timestep]
├── radar/            # Radar rainfall [sample_number, timestep]
├── wet_radar/        # Wet/dry labels [sample_number]
└── cml_rain/         # CML-derived rain rates [sample_number, timestep, channel_id]
```

## 🏗️ Project Structure

```
cml_wd_pytorch/
├── src/cml_wd_pytorch/
│   ├── models/
│   │   └── cnn.py              # CNN model architecture
│   ├── train/
│   │   ├── training_wet_dry.py # Wet/dry classification training
│   │   └── training_rain_rate.py # Rain rate estimation training
│   ├── dataloader/
│   │   └── dataloaderzarr.py   # Zarr dataset loader
│   ├── inference/
│   │   └── run_inference.py    # Inference pipeline
│   ├── evaluation/
│   │   ├── summarize_scores.py # Evaluation utilities
│   │   └── summarize_scores_wet_dry.py
│   └── config/
│       └── config.yml          # Configuration file
├── preprocessing/
│   ├── create_dataset.py       # Dataset creation pipeline
│   ├── cml_radklim_to_zarr.py  # Data format conversion
├── data/
│   ├── dummy_data.zarr         # Example dataset
│   ├── dummy_model             # Example models
│   └── gen_dummy_data.py       # Dummy data generator
├── results/                    # Training outputs and models
├── pyproject.toml              # Project configuration
├── environment.yml             # Conda environment [outdated]
└── LICENSE                     # BSD 3-Clause License
```

## 🎯 Model Architecture

The CNN model features:
- **Input**: 2-channel time series (180 timesteps)
- **Convolutional blocks**: Multi-layer 1D convolutions with ReLU
- **Max pooling**: Temporal dimensionality reduction
- **Fully connected layers**: Dense layers with dropout (40% default)
- **Configurable output**: Sigmoid (classification) or ReLU (regression)

Default architecture:
- Filters: [48, 96, 96, 192, 192]
- Kernel size: 3
- FC neurons: 128
- Dropout: 0.4

## 📈 Performance Metrics

The package provides comprehensive evaluation:

### Classification Metrics
- Accuracy
- True Positive Rate (TPR)
- True Negative Rate (TNR)
- Binary Cross Entropy (BCE) loss

### Regression Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Pearson correlation coefficient

## 🔬 Scientific Background

This implementation is based on the methodology described in:

> Polz, J., et al. (2020). "Rainfall event detection in commercial microwave link attenuation data using convolutional neural networks." *Atmospheric Measurement Techniques*, 13, 3835–3853. [DOI: 10.5194/amt-13-3835-2020](https://doi.org/10.5194/amt-13-3835-2020)

Commercial microwave links (CMLs) are point-to-point radio connections used in cellular networks. Rain causes signal attenuation that can be exploited for precipitation estimation, making CML networks valuable for meteorological applications.

## 📚 Examples

Example data and preprocessing scripts are available in the `data/` and `preprocessing/` directories:
- Data preprocessing workflows
- Model evaluation and analysis tools
- Dummy data generation for testing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Development Setup

```bash
git clone https://github.com/jpolz/cml_wd_pytorch.git
cd cml_wd_pytorch
pip install -e .[dev]
```

## 👥 Contributors

- **Julius Polz** ([@jpolz](https://github.com/jpolz)) - *Main Author* - Karlsruhe Institute of Technology
- **[@waggerle](https://github.com/waggerle)** - *Contributor* 
- **[@cchwala](https://github.com/cchwala)** - *Contributor*

## 📄 License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## 📬 Contact

- Julius Polz - julius.polz@kit.edu
- Karlsruhe Institute of Technology (KIT)
- Institute of Meteorology and Climate Research

## 🔗 Related Publications

If you use this software in your research, please cite:

```bibtex
@article{polz_rain_2020,
	title = {Rain event detection in commercial microwave link attenuation data using convolutional neural networks},
	volume = {13},
	issn = {1867-1381},
	doi = {https://doi.org/10.5194/amt-13-3835-2020},
	number = {7},
	urldate = {2020-12-04},
	journal = {Atmospheric Measurement Techniques},
	author = {Polz, Julius and Chwala, Christian and Graf, Maximilian and Kunstmann, Harald},
	month = jul,
	year = {2020},
	note = {Publisher: Copernicus GmbH},
	pages = {3835--3853},
```
