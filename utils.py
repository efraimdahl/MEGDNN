import h5py
from os import listdir
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def get_dataset_name(file_path):
    file_name = file_path.split('/')[-1]
    temp = file_name.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name


def read_datasets(data_dir):
    datasets = dict()
    datasets['intra'] = {'X_train': [], 'y_train':[], 'X_test': [], 'y_test': []}
    datasets['cross'] ={'X_train': [], 'y_train': [], 'X_test1': [], 'y_test1': [], 'X_test2': [], 'y_test2': [], 'X_test3': [], 'y_test3': []}
    for file in listdir(data_dir+'/Cross'):
        set_type = file
        set_path = data_dir+'/Cross/'+file
        if file != '.DS_Store':
            for file in listdir(set_path):
                file_path = set_path+'/'+file
                with h5py.File(file_path, 'r') as f:
                    matrix = f.get(get_dataset_name(file_path))[()]
                    datasets['cross']['X_'+set_type].append(matrix)
                    label = file.split('_')[:-2]
                    datasets['cross']['y_'+set_type].append('_'.join(label))
    
    for file in listdir(data_dir+'/Intra'):
        set_type = file
        set_path = data_dir+'/Intra/'+file
        if file != '.DS_Store':
            for file in listdir(set_path):
                file_path = set_path+'/'+file
                with h5py.File(file_path, 'r') as f:
                    matrix = f.get(get_dataset_name(file_path))[()]
                    datasets['intra']['X_'+set_type].append(matrix)
                    label = file.split('_')[:-2]
                    datasets['intra']['y_'+set_type].append('_'.join(label))
    
    return datasets


class MEGDataset(Dataset):
    def __init__(self, matrices, labels, transform=None):
        self.matrices = matrices
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.matrices)
    
    def __getitem__(self, idx):
        # read matrix and label, turn to torch tensor, apply transformation if needed
        matrix = torch.from_numpy(self.matrices[idx]).float()
        label = self.labels[idx]
        if self.transform:
            matrix = self.transform(matrix)
        return matrix, label


def fit_transform_scaler(X_train, X_test, scaler):
    # Flatten the training data to fit the scaler
    # Each matrix is reshaped into a single row
    X_train_flattened = np.vstack([x.flatten() for x in X_train])
    scaler.fit(X_train_flattened)

    # Transform the training data
    X_train_scaled = [scaler.transform(x.flatten().reshape(1, -1)).reshape(x.shape) for x in X_train]

    # Transform the test data
    X_test_scaled = []
    for i, test in enumerate(X_test):
        X_test_scaled.append([scaler.transform(x.flatten().reshape(1, -1)).reshape(x.shape) for x in test]
    )

    return X_train_scaled, X_test_scaled


def temporal_downsampling(data, downsample_factor=5):
    """
    Downsamples the time series data by averaging over intervals.

    :param data: The MEG data, assumed to be a list of matrices (samples) with shape [n_sensors, n_timepoints].
    :param downsample_factor: The factor by which to downsample the time series.
    :return: Downsampled MEG data.
    """
    downsampled_data = []
    for sample in data:
        # Reshape the sample if necessary to ensure it's a 2D matrix
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        # Compute the number of time points in the downsampled data
        n_timepoints = sample.shape[1] // downsample_factor

        # Downsample by averaging over intervals
        downsampled_sample = np.mean(sample[:, :n_timepoints * downsample_factor].reshape(sample.shape[0], n_timepoints, downsample_factor), axis=2)
        downsampled_data.append(downsampled_sample)

    return downsampled_data


def batchify_activity(X, y, window_size = 200):
    """
    Creates batches of activity from the MEG data.

    :param X: The MEG data, assumed to be a list of matrices (samples) with shape [n_sensors, n_timepoints].
    :param y: The labels for the MEG data.
    :param window_size: The size of the window to use for each batch.
    :return: X_batched, y_batched: The batched MEG data and labels.
    """
    X_batched = []
    y_batched = []
    for i, sample in enumerate(X):

        # Compute the number of batches that can be created from the sample
        n_batches = sample.shape[1] // window_size

        # Create batches
        for j in range(n_batches):
            X_batched.append(sample[:, j * window_size:(j + 1) * window_size])
            y_batched.append(y[i])

    return X_batched, y_batched 
