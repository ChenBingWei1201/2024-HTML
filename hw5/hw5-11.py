import os
import requests
import bz2
import numpy as np
from liblinear.liblinearutil import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, hstack

# Constants
LAMBDA_VALUES = [0.01, 0.1, 1, 10, 100, 1000]
N_EXPERIMENTS = 1126
TRAIN_SIZE = 8000

# Step 1: Download and decompress the data
def download_and_extract(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {url}...")
        response = requests.get(url)
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {dest_path}.")
    else:
        print(f"{dest_path} already exists.")

def decompress_bz2(file_path, output_path):
    if not os.path.exists(output_path):
        print(f"Decompressing {file_path}...")
        with bz2.BZ2File(file_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"Decompressed to {output_path}.")
    else:
        print(f"{output_path} already exists.")

train_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2"
test_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2"
train_file_compressed = "mnist.scale.bz2"
test_file_compressed = "mnist.scale.t.bz2"
train_file = "mnist.scale"
test_file = "mnist.scale.t"

download_and_extract(train_url, train_file_compressed)
download_and_extract(test_url, test_file_compressed)
decompress_bz2(train_file_compressed, train_file)
decompress_bz2(test_file_compressed, test_file)

# Step 2: Load and preprocess the data
y_train, X_train = svm_read_problem(train_file, return_scipy=True)
y_test, X_test = svm_read_problem(test_file, return_scipy=True)

# Filter for classes 2 and 6, and relabel them for binary classification
train_mask = np.isin(y_train, [2, 6])
test_mask = np.isin(y_test, [2, 6])
y_train, X_train = y_train[train_mask], X_train[train_mask]
y_test, X_test = y_test[test_mask], X_test[test_mask]
y_train = np.where(y_train == 2, 1, -1)
y_test = np.where(y_test == 2, 1, -1)

def align_features(train, test):
    """Ensure train and test sets have the same number of features."""
    if train.shape[1] > test.shape[1]:
        # Pad test set with zeros
        padding = csr_matrix((test.shape[0], train.shape[1] - test.shape[1]))
        test = hstack([test, padding])
    elif test.shape[1] > train.shape[1]:
        # Pad train set with zeros
        padding = csr_matrix((train.shape[0], test.shape[1] - train.shape[1]))
        train = hstack([train, padding])
    return train, test

# Align feature dimensions between training and test sets
X_train, X_test = align_features(X_train, X_test)

# Scale the data using csr_scale
scale_param = csr_find_scale_param(X_train, lower=0)
X_train = csr_scale(X_train, scale_param)
X_test = csr_scale(X_test, scale_param)

# Function to calculate error
def calculate_error(y_true, y_pred):
    return np.mean(y_true != y_pred) * 100

# Prepare for histogram
E_out_values = []

# Main loop for 1126 experiments
for experiment in range(N_EXPERIMENTS):
    np.random.seed(experiment)
    # Step 1: Split data into sub-training and validation sets
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train, y_train, train_size=TRAIN_SIZE, random_state=experiment
    )
    best_lambda, best_E_val = None, float('inf')

    # Step 2: Evaluate each lambda
    for lambda_value in LAMBDA_VALUES:
        C = 1 / lambda_value
        model = train(y_subtrain, X_subtrain, f'-s 6 -c {C} -B 1 -q')
        p_val, _, _ = predict(y_val, X_val, model, '-q')
        E_val = calculate_error(y_val, p_val)
        if E_val < best_E_val or (E_val == best_E_val and lambda_value > best_lambda):
            best_E_val = E_val
            best_lambda = lambda_value

    # Step 3: Re-train with the best lambda on the full training set
    C = 1 / best_lambda
    final_model = train(y_train, X_train, f'-s 6 -c {C} -B 1 -q')
    
    # Step 4: Evaluate E_out on the test set
    p_test, _, _ = predict(y_test, X_test, final_model, '-q')
    E_out = calculate_error(y_test, p_test)
    E_out_values.append(E_out)

# Step 5: Plot histogram of E_out
plt.hist(E_out_values, bins=20, edgecolor='black', alpha=0.7)
plt.title('Histogram of $E_{out}$')
plt.xlabel('$E_{out}$ (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
