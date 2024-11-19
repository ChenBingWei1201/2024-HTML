'''
10. Consider L1-regularized logistic regression w_lambda = argmin_w lambda/N*||w||_1 + 1/N*sum_{n=1}^N ln(1 + exp(-y_n*w^T*x_n)), where x_n and w are (d + 1) dimensional vectors, as usual. That is, each x is padded with x_0 = 1. We will use the mnist.scale dataset at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2 for training, and https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2 for testing (evaluating E_out). The datasets are originally for multi-class classification. We will study one of the sub-problems within one-versus-one decomposition: class 2 versus class 6.

We call the algorithm for solving the problem above as A_lambda . The problem guides you to use LIBLINEAR (https://www.csie.ntu.edu.tw/~cjlin/liblinear/), a machine learning packaged developed at National Taiwan University, to solve this problem. In addition to using the default options, what you need to do when running LIBLINEAR are
- set option `-s 6`, which corresponds to solving L1-regularized logistic regression
- set option `-c C`, with a parameter value of `C` calculated from the lambda that you want to use; read README of the software package to figure out how C and your lambda should relate to each other.

LIBLINEAR can be called from the command line or from major programming languages like python. You may find other options of the package useful for some of the problems below. It is up to you to decide whether to use them or not after checking the README. After extracting the data of only 2 and 6 from the training and test sets, we will consider the dataset as a binary classification problem and take the “regression for classification” approach with regularized logistic regression (see Page 6 of Lecture 10). So please evaluate all errors below with the 0/1 error.

First select the best lambda* as argmin_{log_{10}lambda ∈ {-2,-1,0,1,2,3}} E_in(w_lambda). Break the tie, if any, by selecting the largest lambda. Note that `-s 6` is a randomized solver that may return different results on different random seeds, even with the same data. Repeat the experiments above for 1126 times, each with a different random seed. Plot a histogram of E_out(g), where g is returned by running A_lambda with lambda*, over the 1126 experiments. In addition, plot a histogram of the average number of non-zero components in each g.
'''
import os
import requests
import bz2
import numpy as np
import scipy.sparse
from liblinear.liblinearutil import *
import matplotlib.pyplot as plt

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
        padding = scipy.sparse.csr_matrix((test.shape[0], train.shape[1] - test.shape[1]))
        test = scipy.sparse.hstack([test, padding])
    elif test.shape[1] > train.shape[1]:
        # Pad train set with zeros
        padding = scipy.sparse.csr_matrix((train.shape[0], test.shape[1] - train.shape[1]))
        train = scipy.sparse.hstack([train, padding])
    return train, test

# Align feature dimensions between training and test sets
X_train, X_test = align_features(X_train, X_test)

# Scale the data using csr_scale
scale_param = csr_find_scale_param(X_train, lower=0)
X_train = csr_scale(X_train, scale_param)
X_test = csr_scale(X_test, scale_param)

# Step 3: Define helper functions
def lambda_to_C(lmbda):
    return 1 / lmbda

lambdas = [0.01, 0.1, 1, 10, 100, 1000]
N = len(y_train)
best_lambda, min_error = None, float('inf')
errors_in = []

# Step 4: Find the best lambda and repeat experiments with 1126 seeds
random_seeds = range(1126)
errors_out, non_zero_counts = [], []

for seed in random_seeds:
    for lmbda in lambdas:
        C = lambda_to_C(lmbda)
        model = train(y_train, X_train, f'-s 6 -c {C} -B 1 -q')
        _, p_acc, _ = predict(y_train, X_train, model)
        E_in = 100 - p_acc[0]
        errors_in.append(E_in)
        if E_in < min_error or (E_in == min_error and lmbda > best_lambda):
            best_lambda, min_error = lmbda, E_in
    
    C = lambda_to_C(best_lambda)
    np.random.seed(seed)
    model = train(y_train, X_train, f'-s 6 -c {C} -B 1 -q')
    _, p_acc, _ = predict(y_test, X_test, model)
    E_out = 100 - p_acc[0] # accuracy to error
    errors_out.append(E_out)

    # Access weights and bias terms
    weights, bias = model.get_decfun()
    non_zero_counts.append(np.sum(np.abs(weights) > 1e-6))

# Step 6: Plot histograms
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(errors_out, bins=30, color="blue", edgecolor="black", alpha=0.7)
plt.title("Histogram of $E_{out}(g)$")
plt.xlabel("$E_{out}$ (%)")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(non_zero_counts, bins=30, color="green", edgecolor="black", alpha=0.7)
plt.title("Histogram of Non-Zero Components in $g$")
plt.xlabel("Number of Non-Zero Components")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
