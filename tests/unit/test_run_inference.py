"""
Unit tests for inference functions.
"""

import pytest
import torch
from unittest.mock import patch, mock_open
from pathlib import Path

from cml_wd_pytorch.inference.run_inference import set_device, load_config


class TestSetDevice:
    """Test the set_device function."""
    
    def test_set_device_cuda_available(self):
        """Test set_device returns CUDA device when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            device = set_device()
            assert device.type == 'cuda'
            assert isinstance(device, torch.device)
    
    def test_set_device_cuda_not_available(self):
        """Test set_device returns CPU device when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            device = set_device()
            assert device.type == 'cpu'
            assert isinstance(device, torch.device)
    
    def test_set_device_returns_torch_device(self):
        """Test that set_device always returns a torch.device object."""
        device = set_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'cpu']


class TestLoadConfig:
    """Test the load_config function."""
    
    @patch('builtins.open', new_callable=mock_open, read_data="""
data:
  reflength: 60
  batch_size: 32
training:
  epochs: 100
  learning_rate: 0.001
""")
    @patch('os.path.abspath')
    def test_load_config_success(self, mock_abspath, mock_file):
        """Test successful config loading."""
        # Mock the path resolution
        mock_abspath.return_value = '/fake/path/to/run_inference.py'
        
        config = load_config()
        
        # Verify the config was parsed correctly
        assert config['data']['reflength'] == 60
        assert config['data']['batch_size'] == 32
        assert config['training']['epochs'] == 100
        assert config['training']['learning_rate'] == 0.001
    
    @patch('builtins.open', side_effect=FileNotFoundError("Config file not found"))
    @patch('os.path.abspath')
    def test_load_config_file_not_found(self, mock_abspath, mock_file):
        """Test that FileNotFoundError is raised when config file doesn't exist."""
        mock_abspath.return_value = '/fake/path/to/run_inference.py'
        
        with pytest.raises(FileNotFoundError):
            load_config()
    
    @patch('builtins.open', new_callable=mock_open, read_data="invalid: yaml: content: [")
    @patch('os.path.abspath')
    def test_load_config_invalid_yaml(self, mock_abspath, mock_file):
        """Test that YAML parsing errors are handled."""
        mock_abspath.return_value = '/fake/path/to/run_inference.py'
        
        with pytest.raises(Exception):  # yaml.safe_load will raise an exception for invalid YAML
            load_config()