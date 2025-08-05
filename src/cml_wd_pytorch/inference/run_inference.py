"""
Takes an xarray dataarray and runs inference on it using a PyTorch model.
Workflow:
1. Load the model.
2. Prepare the data.
3. Run inference in batches.
4. Collect and return predictions.

Dataarray shape is expected to be (time, channels, cml_id).
Output shape will be (time, channels, cml_id) with predictions for each time step.
Model input is of shape (batch_size, channels, time_window), where target time and
cml_id are captured in the batch.

"""

import numpy as np
import torch
import xarray as xr

from cml_wd_pytorch.models.cnn import cnn


def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def predict_batch(model, batch, device):
    model.eval()
    with torch.no_grad():
        inputs = batch.to(device)
        outputs = model(inputs)
    return outputs


def load_model(model_path, device):
    """
    Loads a PyTorch model from the specified path.
    Args:
        model_path (str): Path to the model file.
        device (torch.device): Device to load the model on.
    Returns:
        model (torch.nn.Module): The loaded PyTorch model.
    """
    # Create the model instance first
    model = cnn(
        final_act="sigmoid"
    )  # Default to sigmoid, might need to be configurable

    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move model to device
    model.to(device)

    # Add window_size attribute (based on the data preprocessing, it's 180)
    model.window_size = 180

    return model


def rolling_window(timeseries, valid_times, window_size):
    """
    Splits the time series into batches of specified size.
    Args:
        timeseries (list or np.array): The time series data to be split.
        valid_times (list): A list of valid time indices.
        window_size (int): The size of each batch.
    Returns:
        windowed_series (np.array): A list of batches, each containing a segment of the time series.
        timestep_indices (list): A list of indices corresponding to the start of each batch.
    """
    assert window_size > 0, "Window size must be greater than 0"
    assert len(valid_times) == len(timeseries), (
        "Valid times must match the length of the timeseries"
    )
    windowed_series = []
    timestep_indices = []
    for start in range(len(timeseries)):
        end = start + window_size
        if end <= len(timeseries):
            windowed_series.append(timeseries[start:end])
            timestep_indices.append(valid_times[start])
    return windowed_series, timestep_indices


def batchify_windows(data, window_size, batch_size):
    """
    Converts a data array into batches of specified window size.
    Args:
        data (xarray.DataArray): The input data array.
        window_size (int): The size of each time series window.
        batch_size (int): The number of samples in each batch.
    Returns:
        combined_samples (dict): A dictionary containing concatenated cml_id, time, and data arrays.
    """
    samples = []
    cml_ids = data.cml_id.values
    # iterate over cml_id dimension
    for cml in cml_ids:
        cml_data = data.sel(cml_id=cml)
        # iterate over time dimension
        timeseries = cml_data.values
        valid_times = cml_data.time.values
        windowed_series, timestep_indices = rolling_window(
            timeseries, valid_times, window_size
        )
        cml_id = np.repeat(cml, len(windowed_series))
        samples.append(
            {
                "cml_id": cml_id,
                "time": timestep_indices,
                "data": np.array(windowed_series),
            }
        )
    # Combine all samples into a single array
    combined_samples = {
        "cml_id": np.concatenate([sample["cml_id"] for sample in samples]),
        "time": np.concatenate([sample["time"] for sample in samples]),
        "data": np.concatenate([sample["data"] for sample in samples]),
    }

    return combined_samples


def build_dataloader(data, window_size, batch_size, device):
    """
    Builds a PyTorch DataLoader from the input data.
    Args:
        data (xarray.DataArray): The input data array.
        window_size (int): The size of each time series window.
        batch_size (int): The number of samples in each batch.
        device (torch.device): The device to run the model on.
    Returns:
        dataloader (torch.utils.data.DataLoader): A DataLoader for the input data.
    """
    combined_samples = batchify_windows(data, window_size, batch_size)

    # Convert data to the right format for the model
    # Data should be (batch_size, channels, time_window)
    tensor_data = torch.tensor(combined_samples["data"], dtype=torch.float32)

    # The model expects (batch_size, channels, time_window), so we need to transpose
    # from (batch_size, time_window, channels) to (batch_size, channels, time_window)
    tensor_data = tensor_data.permute(0, 2, 1)

    dataset = torch.utils.data.TensorDataset(
        tensor_data,
        torch.tensor(combined_samples["cml_id"], dtype=torch.long),
        torch.tensor(combined_samples["time"], dtype=torch.long),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    return dataloader


def run_inference(model, data, batch_size=32):
    device = set_device()
    window_size = (
        model.window_size if hasattr(model, "window_size") else 180
    )  # TODO: remove hardcoded value
    dataloader = build_dataloader(data, window_size, batch_size, device)
    predictions = []
    cml_ids_list = []
    times_list = []

    for batch in dataloader:
        inputs, cml_ids, times = batch
        inputs = inputs.to(device)
        outputs = predict_batch(model, inputs, device)
        predictions.append(outputs.cpu())
        cml_ids_list.append(cml_ids)
        times_list.append(times)

    # Concatenate all predictions
    all_predictions = torch.cat(predictions, dim=0)
    all_cml_ids = torch.cat(cml_ids_list, dim=0)
    all_times = torch.cat(times_list, dim=0)

    return {"predictions": all_predictions, "cml_ids": all_cml_ids, "times": all_times}


def redistribute_results(results, data):
    """
    Redistribute the 1D inference results back to the original data structure.

    Args:
        results (dict): Dictionary containing predictions, cml_ids, and times from inference
        data (xarray.Dataset): Original dataset to add predictions to

    Returns:
        xarray.Dataset: Dataset with predictions added as a new variable
    """
    predictions = (
        results["predictions"].numpy().squeeze()
    )  # Remove any extra dimensions
    cml_ids = results["cml_ids"].numpy()
    times = results["times"].numpy()

    # Get original dimensions
    ref_times = data.time.to_numpy()
    ref_cml_ids = data.cml_id.to_numpy()

    # Initialize prediction array with NaN values
    pred_array = np.full((len(ref_times), len(ref_cml_ids)), np.nan)

    # Map predictions back to the original grid
    for i, (pred_cml_id, pred_time, pred_value) in enumerate(
        zip(cml_ids, times, predictions)
    ):
        # Find indices in the original data
        try:
            cml_idx = np.where(ref_cml_ids == pred_cml_id)[0][0]
            time_idx = np.where(ref_times == pred_time)[0][0]
            pred_array[time_idx, cml_idx] = pred_value
        except (IndexError, ValueError):
            # Skip if the time or cml_id is not found in the original data
            continue

    # Create a new DataArray to hold the predictions
    pred_data = xr.DataArray(
        pred_array,
        dims=["time", "cml_id"],
        coords={
            "time": ref_times,
            "cml_id": ref_cml_ids,
        },
        name="predictions",
    )

    # Add the predictions to the dataset
    return data.assign(predictions=pred_data)


def cnn_wd(model_path, data, batch_size=32):
    """
    Function to run wet/dry inference on input data using a trained CNN model.
    Args:
        model_path (str): Path to the trained PyTorch model.
        data (xarray.DataArray): The input data array.
        batch_size (int): The number of samples in each batch.
    Returns:
        xarray.Dataset: Dataset with predictions added as a new variable.
    """
    device = set_device()
    model = load_model(model_path, device)
    results = run_inference(model, data, batch_size)
    data = data.to_dataset(name="TL")  # Convert xarray DataArray to Dataset if needed
    final_results = redistribute_results(results, data)
    return final_results


def test_cnn_wd():
    """
    Test function to run inference with a sample model and data.
    This is for demonstration purposes and should be replaced with actual data and model paths.
    """
    # Example usage
    model_path = "/bg/fast/env_polz-j/uvprojects/cml_wd_pytorch/data/dummy_model/model_epoch_0.pth"  # Replace with your model path
    data = xr.DataArray(
        np.random.rand(1000, 2, 5),
        dims=["time", "channels", "cml_id"],
        coords={
            "time": np.arange(1000),
            "channels": np.arange(2),
            "cml_id": np.arange(5),
        },
    )
    final_dataset = cnn_wd(model_path, data, batch_size=32)
    import logging

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Final dataset variables: {list(final_dataset.data_vars.keys())}")
    logging.info(f"Final dataset dimensions: {final_dataset.dims}")
    if "predictions" in final_dataset:
        logging.info(f"Predictions shape: {final_dataset['predictions'].shape}")
        logging.info(f"Predictions dimensions: {final_dataset['predictions'].dims}")
        logging.info(
            f"Sample predictions:\n{final_dataset['predictions'][:5, :3].values}"
        )


def main():
    """
    Main function for running inference as a standalone script.
    Creates sample data and runs inference using command line arguments.
    """
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Run inference on a PyTorch model with xarray data."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained PyTorch model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference."
    )

    args = parser.parse_args()

    # Create sample data for demonstration
    data = xr.DataArray(
        np.random.rand(1000, 2, 5),
        dims=["time", "channels", "cml_id"],
        coords={
            "time": np.arange(1000),
            "channels": np.arange(2),
            "cml_id": np.arange(5),
        },
    )

    predictions = cnn_wd(args.model_path, data, args.batch_size)
    logging.info(f"Inference completed. Predictions: {predictions}")


if __name__ == "__main__":
    main()

    # test_main()  # Run the test function to demonstrate functionality
