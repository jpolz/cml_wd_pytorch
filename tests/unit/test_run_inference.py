"""
Unit tests for inference functions from cml_wd_pytorch.inference.run_inference.

Test Coverage Overview:
=======================

TestSetDevice:
    - CUDA device detection when CUDA is available
    - CPU fallback when CUDA is not available
    - Return type validation (torch.device)
    - Cross-platform compatibility

TestLoadConfig:
    - Successful YAML configuration file loading
    - File path resolution and error handling
    - YAML parsing validation
    - FileNotFoundError handling for missing config files
    - Invalid YAML syntax error handling
    - Configuration structure validation

Functions Under Test:
====================
- set_device(): Device selection logic for CPU/CUDA inference
- load_config(): Configuration file loading and parsing

Test Patterns Used:
==================
- Mocking external dependencies (torch.cuda, file operations)
- Parameterized testing concepts for different scenarios
- Exception testing for error conditions
- Return type and value validation
- Path mocking for cross-platform compatibility

Dependencies Mocked:
===================
- torch.cuda.is_available: For CUDA availability testing
- builtins.open: For file I/O operations
- os.path.abspath: For path resolution
- yaml.safe_load: Implicitly tested through file content mocking
"""

from unittest.mock import mock_open, patch

import pytest
import torch

from cml_wd_pytorch.inference.run_inference import load_config, set_device


class TestSetDevice:
    """Test the set_device function."""

    def test_set_device_cuda_available(self):
        """Test set_device returns CUDA device when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            device = set_device()
            assert device.type == "cuda"
            assert isinstance(device, torch.device)

    def test_set_device_cuda_not_available(self):
        """Test set_device returns CPU device when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            device = set_device()
            assert device.type == "cpu"
            assert isinstance(device, torch.device)

    def test_set_device_returns_torch_device(self):
        """Test that set_device always returns a torch.device object."""
        device = set_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "cpu"]


class TestLoadConfig:
    """Test the load_config function."""

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="""
data:
  reflength: 60
  batch_size: 32
training:
  epochs: 100
  learning_rate: 0.001
""",
    )
    @patch("os.path.abspath")
    def test_load_config_success(self, mock_abspath, mock_file):
        """Test successful config loading."""
        # Mock the path resolution
        mock_abspath.return_value = "/fake/path/to/run_inference.py"

        config = load_config()

        # Verify the config was parsed correctly
        assert config["data"]["reflength"] == 60
        assert config["data"]["batch_size"] == 32
        assert config["training"]["epochs"] == 100
        assert config["training"]["learning_rate"] == 0.001

    @patch("builtins.open", side_effect=FileNotFoundError("Config file not found"))
    @patch("os.path.abspath")
    def test_load_config_file_not_found(self, mock_abspath, mock_file):
        """Test that FileNotFoundError is raised when config file doesn't exist."""
        mock_abspath.return_value = "/fake/path/to/run_inference.py"

        with pytest.raises(FileNotFoundError):
            load_config()

    @patch(
        "builtins.open", new_callable=mock_open, read_data="invalid: yaml: content: ["
    )
    @patch("os.path.abspath")
    def test_load_config_invalid_yaml(self, mock_abspath, mock_file):
        """Test that YAML parsing errors are handled."""
        mock_abspath.return_value = "/fake/path/to/run_inference.py"

        with pytest.raises(
            Exception
        ):  # yaml.safe_load will raise an exception for invalid YAML
            load_config()
