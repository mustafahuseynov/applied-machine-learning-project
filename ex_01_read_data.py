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
    data = data[data['labels'] != -1].copy()

    data.dropna(inplace=True)

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

    labels = data['labels'].to_numpy()
    exp_ids = data['exp_ids'].to_numpy()

    features_columns = [f'I_{i:03d}' for i in range(200)] + [f'V_{i:03d}' for i in range(200)]
    features = data[features_columns].to_numpy()

    return labels, exp_ids, features

def create_sliding_windows_first_dim(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Create sliding windows over the first dimension of a 3D array.
    
    Args:
        data (np.ndarray): Input array of shape (n_samples, timesteps, features)
        sequence_length (int): Length of each window
    
    Returns:
        np.ndarray: Windowed data of shape (n_windows, sequence_length*timesteps, features)
    """

    n_samples, timesteps, features = data.shape

    n_windows = n_samples - sequence_length + 1

    windowed_data_list = []

    for i in range(n_windows):
        current_window = data[i : i + sequence_length, :, :]

        reshaped_window = current_window.reshape(sequence_length * timesteps, features)
        
        windowed_data_list.append(reshaped_window)

    return np.array(windowed_data_list)

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
    cache_dir = path.parent
    file_stem = path.stem
    labels_cache_path = cache_dir / f"{file_stem}_labels.npy"
    exp_ids_cache_path = cache_dir / f"{file_stem}_exp_ids.npy"
    features_cache_path = cache_dir / f"{file_stem}_features.npy"

    if labels_cache_path.is_file() and exp_ids_cache_path.is_file() and features_cache_path.is_file():
        print(f"Loading data from cache: {labels_cache_path}, {exp_ids_cache_path}, {features_cache_path}")
        labels = np.load(labels_cache_path)
        exp_ids = np.load(exp_ids_cache_path)
        features = np.load(features_cache_path)
    else:
        print(f"Cache files not found. Loading data from CSV: {path}")
        
        df = load_data(path)
        
        labels, exp_ids, features = convert_to_np(df)

        np.save(labels_cache_path, labels)
        np.save(exp_ids_cache_path, exp_ids)
        np.save(features_cache_path, features)
        print("Data saved to cache.")

    if return_sequences:
        print(f"Creating sliding windows with sequence length: {sequence_length}")

        current_features = features[:, :200]
        voltage_features = features[:, 200:]
        
        features_reshaped = np.stack((current_features, voltage_features), axis=-1)

        features = create_sliding_windows_first_dim(features_reshaped, sequence_length)
        
        labels = labels[sequence_length - 1:]
        exp_ids = exp_ids[sequence_length - 1:]

    if n_samples is not None and n_samples < len(labels):
        print(f"Sampling {n_samples} samples from the data.")

        indices = np.random.choice(len(labels), n_samples, replace=False)

        labels = labels[indices]
        exp_ids = exp_ids[indices]
        features = features[indices]
    elif n_samples is not None and n_samples >= len(labels):
        print(f"Requested n_samples ({n_samples}) is greater than or equal to available data ({len(labels)}). Returning all available data.")


    return features, labels, exp_ids