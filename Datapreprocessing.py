import h5py
import os
from sklearn.preprocessing import MinMaxScaler

def get_dataset_name(file_path):
    file_name = file_path.split('/')[-1]
    temp = file_name.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

base_directory = "./Final Project data/Intra/train/"
task_types = ['rest', 'task_motor', 'task_story_math', 'task_working_memory']

scaler = MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)

# Loop over task types
for task_type in task_types:  # Add more task types as needed
    for number in ['1', '2', '3', '4', '5', '6', '7', '8']:  # Add more numbers as needed
        # Construct the file path
        file_path = os.path.join(base_directory, f"{task_type}_105923_{number}.h5")

        # Process the file
        with h5py.File(file_path, 'r') as f:
            dataset_name = get_dataset_name(f ile_path)
            matrix = f.get(dataset_name)[()]
            # Reshape the matrix to a 1D array if needed

            # Apply MinMaxScaler
            scaled_data = scaler.fit_transform(matrix)

            # Print information about the scaled data
            print(f"File: {file_path}")
            print(f"Scaled data type: {type(scaled_data)}")
            print(f"Scaled data shape: {scaled_data.shape}")

            # Print the first few instances of the scaled data
            print("First few instances of the scaled data:")
            print(scaled_data[:5, :5])
            print()