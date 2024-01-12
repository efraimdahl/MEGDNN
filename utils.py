import h5py
from os import listdir
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import numpy as np


def get_dataset_name(file_path):
    """
    Extracts the dataset name from a given file path.

    :param file_path: The complete file path, including the file name.
    :return: dataset_name: The extracted dataset name, obtained by removing the last part of the file name and joining the remaining parts with underscores.
    """
    file_name = file_path.split('/')[-1]
    temp = file_name.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name


def read_datasets(data_dir):
    """
    Reads the datasets form a given directory.
    For both intra and cross, it extracts the datasets and labels.

    :param data_dir: The path to the directory containing dataset files.
    :return: datasets: A dictionary containing intra and cross datasets, each with train and test sets.

    """
    datasets = dict()
    datasets['intra'] = {'X_train': [], 'y_train':[], 'X_test': [], 'y_test': []}
    datasets['cross'] ={'X_train': [], 'y_train': [], 'X_test1': [], 'y_test1': [], 'X_test2': [], 'y_test2': [], 'X_test3': [], 'y_test3': []}
    
    # process Cross datasets
    for file in listdir(data_dir+'/Cross'):
        set_type = file
        set_path = data_dir+'/Cross/'+file
        if file != '.DS_Store':
            for file in listdir(set_path):
                file_path = set_path+'/'+file
                with h5py.File(file_path, 'r') as f:
                    matrix = f.get(get_dataset_name(file_path))[()] # extracts dataset matrix
                    datasets['cross']['X_'+set_type].append(matrix)
                    label = file.split('_')[:-2] # extracts dataset label
                    datasets['cross']['y_'+set_type].append('_'.join(label))
    
    # process Intra datasets
    for file in listdir(data_dir+'/Intra'):
        set_type = file
        set_path = data_dir+'/Intra/'+file
        if file != '.DS_Store':
            for file in listdir(set_path):
                file_path = set_path+'/'+file
                with h5py.File(file_path, 'r') as f:
                    matrix = f.get(get_dataset_name(file_path))[()] # extracts dataset matrix
                    datasets['intra']['X_'+set_type].append(matrix)
                    label = file.split('_')[:-2] # extracts dataset label
                    datasets['intra']['y_'+set_type].append('_'.join(label))
    
    return datasets


class MEGDataset(Dataset):
    def __init__(self, matrices, labels, transform=None):
        """
        Custom PyTorch dataset for MEG data.

        :param matrices: List of numpy arrays representing MEG matrices.
        :param labels: List of corresponding labels for each matrix.
        :param transform: A transformation to be applied to the matrix data.
        """
        self.matrices = matrices
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        # returns the number of samples
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


def error_analysis(model, data, encoder_dict, config):
    '''
    Performs error analysis on each cross test set 
    and calculates the missclassification rate for each class and the percentages for which classes they are missclassified as.
    '''
    decoder_dict = {v: k for k, v in encoder_dict.items()}
    # iterate over the three cross test sets and perform error analysis for each subject
    for i in ['1', '2', '3']:
        X = data['cross']['X_test'+i]
        y = data['cross']['y_test'+i]
        y = [encoder_dict[label] for label in y]
        X_scaled = fit_transform_scaler(data['cross']['X_train'], [X], scaler=StandardScaler())[1][0]
        X = temporal_downsampling(X_scaled, downsample_factor=config['downsample'])
        if config['window_size'] != -1:
            X, y = batchify_activity(X, y, window_size = config['window_size'])
        if config['model'] != 'MEGConvNet':
            X = [x.flatten() for x in X]
        test_set = MEGDataset(X, y)
        test_loader = DataLoader(test_set, batch_size=config['batch_size'])
        model.eval()
        preds = []
        for inputs, labels in test_loader:
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            with torch.no_grad():
                output = model(inputs)
                preds.extend(output.argmax(dim=1).tolist())
        acc = accuracy_score(y, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average='macro')
        print(f'Subject {i}:\n')
        print(f'Accuracy: {acc}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')
        y = [decoder_dict[label] for label in y]
        preds = [decoder_dict[label] for label in preds]

        # calculates missclassification rate for each class and percentages for which classes they are missclassified as
        for label in encoder_dict.keys():
            print(f'Class {label}:')
            missclassified = [preds[i] for i in range(len(preds)) if preds[i] != y[i] and y[i] == label]
            print(f'Missclassification rate: {len(missclassified)/y.count(label)}')
            print(f'Missclassified as:')
            for miss in encoder_dict.keys():
                if miss != label and len(missclassified) > 0:
                    print(f'{miss}: {missclassified.count(miss)/len(missclassified)}')
            print('\n')
        print('\n')