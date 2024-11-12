import numpy as np
import random
import matplotlib.pyplot as plt
import requests
import bz2

# Define constants
N, num_experiments = 200, 1000
max_consecutive_correct = 5 * N

# Download and prepare the data
def download_and_parse_data():
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2"
    response = requests.get(url)
    filename = "rcv1_train.binary.bz2"
    with open(filename, 'wb') as f:
        f.write(response.content)
    
    # Decompress and read the first 200 lines
    with bz2.open(filename, 'rt') as f:
        lines = [next(f) for _ in range(200)]
    
    return lines

# Preprocess data
def parse_libsvm_data(lines):
    data = []
    max_feature_index = 0
    for line in lines:
        parts = line.split()
        label = int(parts[0])
        features = {int(k): float(v) for k, v in (p.split(":") for p in parts[1:])}
        max_feature_index = max(max_feature_index, max(features.keys(), default=0))
        data.append((label, features))
    return data, max_feature_index

# Convert to dense representation
def convert_to_dense(parsed_data, num_features):
    X = np.zeros((len(parsed_data), num_features + 1))
    y = np.zeros(len(parsed_data))
    
    for i, (label, features) in enumerate(parsed_data):
        y[i] = label
        X[i, 0] = 1  # Bias term
        for index, value in features.items():
            X[i, index] = value
    
    return X, y

# PLA algorithm
def sign(x):
    if x > 0:
        return 1  
    else:
        return -1


def perceptron_learning_algorithm(X, y):
    num_samples, num_features = X.shape
    w = np.zeros(num_features)  # Initialize weights to zero
    consecutive_correct = 0
    updates = 0
    w_norms = []

    # Passing 5N times of correctness checking means that w_t is mistake-free with more than 99% of probability
    while consecutive_correct < max_consecutive_correct:
        # Randomly pick an example
        i = random.randint(0, num_samples - 1)
        if sign(np.dot(w, X[i])) != y[i]:
            w += y[i] * X[i] # Update weights
            updates += 1
            consecutive_correct = 0  # Reset consecutive correct counter
            w_norms.append(np.linalg.norm(w))
        else:
            consecutive_correct += 1

    return updates, w_norms

# Run experiments for 1000 times
def run_experiments(X, y, num_experiments):
    update_counts = []
    all_w_norms = []
    for seed in range(num_experiments):
        random.seed(seed)
        updates, w_norms = perceptron_learning_algorithm(X, y)
        update_counts.append(updates)
        all_w_norms.append(w_norms)
    return update_counts, all_w_norms

# Plot the norm of w as a function of t
def plot_w_norms(all_w_norms, T_min):
    for w_norms in all_w_norms:
        plt.plot(range(1, T_min + 1), w_norms[:T_min], alpha=0.5)
    plt.title("Norm of w as a function of t")
    plt.xlabel("t")
    plt.ylabel("||w_t||")
    plt.show()

def main():
    lines = download_and_parse_data()
    parsed_data, max_feature_index = parse_libsvm_data(lines)
    X, y = convert_to_dense(parsed_data, max_feature_index)
    update_counts, all_w_norms = run_experiments(X, y, 1000)
    T_min = min(update_counts)
    plot_w_norms(all_w_norms, T_min)

if __name__ == "__main__":
    main()