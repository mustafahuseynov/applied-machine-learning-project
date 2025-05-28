import numpy as np
import pandas as pd  
from pathlib import Path


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file. Remove rows with unlabeled data.

    Args:
        data_path (Path): Path to the CSV data file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with unlabeled data removed.
    Raises:
        FileNotFoundError: If the specified data file does not exist.
        ValueError: If the data is empty after removing unlabeled data and dropping NaN values.
    """

    if not data_path.is_file():
        raise FileNotFoundError(f"Data on path {data_path} not found!")

    data = remove_unlabeled_data(pd.read_csv(data_path))

    if data.empty:
        raise ValueError("Data is empty after removing unlabeled data and dropping NaN values!")

    return data

def remove_unlabeled_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with unlabeled data (where labels == -1).

    Args:
        data (pd.DataFrame): Input DataFrame containing a 'labels' column.

    Returns:
        pd.DataFrame: DataFrame with unlabeled data removed.
    """
    data[data['labels' != -1]].copy()

    data.dropna

    return data


def convert_to_np(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DataFrame to numpy arrays, separating labels, experiment IDs, and features.

    Args:
        data (pd.DataFrame): Input DataFrame containing 'labels', 'exp_ids', and feature columns.

    Returns:
        tuple: A tuple containing:
            - labels (np.ndarray): Array of labels
            - exp_ids (np.ndarray): Array of experiment IDs
            - data (np.ndarray): Combined array of current and voltage features
    """
    pass


def create_sliding_windows_first_dim(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Create sliding windows over the first dimension of a 3D array.
    
    Args:
        data (np.ndarray): Input array of shape (n_samples, timesteps, features)
        sequence_length (int): Length of each window
    
    Returns:
        np.ndarray: Windowed data of shape (n_windows, sequence_length*timesteps, features)
    """
    pass

def get_welding_data(path: Path, n_samples: int | None = None, return_sequences: bool = False, sequence_length: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load welding data from CSV or cached numpy files.

    If numpy cache files don't exist, loads from CSV and creates cache files.
    If cache files exist, loads directly from them.

    Args:
        path (Path): Path to the CSV data file.
        n_samples (int | None): Number of samples to sample from the data. If None, all data is returned.
        return_sequences (bool): If True, return sequences of length sequence_length.
        sequence_length (int): Length of sequences to return.
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of welding data features
            - np.ndarray: Array of labels
            - np.ndarray: Array of experiment IDs
    """
    pass
