import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import os
import requests

# Define constants and helper functions
train_data_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/madelon'
test_data_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/madelon.t'

def download_and_extract(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {url} to {dest_path}...")
        response = requests.get(url)
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print("Download complete!")
    else:
        print(f"File already exists at {dest_path}.")

# Load LIBSVM dataset
def load_libsvm_data(file_path):
    """
    Load data in LIBSVM format and convert it to dense format.
    """
    data, labels = load_svmlight_file(file_path)
    X = csr_matrix.toarray(data)  # Convert sparse to dense matrix
    y = np.where(labels == 0, -1, labels)  # Ensure labels are {+1, -1}
    return X, y

# Optimized decision stump implementation
def optimized_decision_stump(X, y, weights):
    """
    Find the best decision stump for the given weights, features, and labels.
    Optimized for performance.
    """
    n_samples, n_features = X.shape
    best_stump = {
        'feature': None,
        'threshold': None,
        'sign': None,
        'error': float('inf')
    }

    for feature in range(n_features):
        sorted_indices = np.argsort(X[:, feature])
        sorted_X = X[sorted_indices, feature]
        sorted_y = y[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cumulative_positive = np.cumsum(sorted_weights * (sorted_y == 1))
        cumulative_negative = np.cumsum(sorted_weights * (sorted_y == -1))
        total_positive = cumulative_positive[-1]
        total_negative = cumulative_negative[-1]

        errors_left = cumulative_negative + (total_positive - cumulative_positive)
        errors_right = cumulative_positive + (total_negative - cumulative_negative)

        for errors, sign in [(errors_left, 1), (errors_right, -1)]:
            min_error_idx = np.argmin(errors)
            if errors[min_error_idx] < best_stump['error']:
                best_stump['feature'] = feature
                best_stump['threshold'] = sorted_X[min_error_idx]
                best_stump['sign'] = sign
                best_stump['error'] = errors[min_error_idx]

    return best_stump

# AdaBoost implementation
def adaboost_with_generalization(X_train, y_train, X_test, y_test, T):
    """
    AdaBoost algorithm with decision stumps and generalization error tracking.
    """
    n_samples, n_features = X_train.shape
    weights = np.ones(n_samples) / n_samples
    alpha = []
    stumps = []

    ein_aggregate = []
    eout_aggregate = []

    for t in range(T):
        # Train a decision stump
        stump = optimized_decision_stump(X_train, y_train, weights)
        predictions = stump['sign'] * np.sign(X_train[:, stump['feature']] - stump['threshold'])
        error = np.sum(weights * (predictions != y_train))

        # Calculate alpha
        alpha_t = 0.5 * np.log((1 - error) / (error + 1e-10))
        alpha.append(alpha_t)
        stumps.append(stump)

        # Update weights
        weights *= np.exp(-alpha_t * y_train * predictions)
        weights /= np.sum(weights)

        # Compute G_t for Ein and Eout
        G_train = np.sign(sum(alpha[i] * (stumps[i]['sign'] *
                                         np.sign(X_train[:, stumps[i]['feature']] - stumps[i]['threshold']))
                              for i in range(len(stumps))))
        G_test = np.sign(sum(alpha[i] * (stumps[i]['sign'] *
                                         np.sign(X_test[:, stumps[i]['feature']] - stumps[i]['threshold']))
                             for i in range(len(stumps))))

        ein = np.mean(G_train != y_train)
        eout = np.mean(G_test != y_test)

        ein_aggregate.append(ein)
        eout_aggregate.append(eout)

    return ein_aggregate, eout_aggregate

# Plotting the results
def plot_generalization_error(ein, eout):
    """
    Plot Ein(G_t) and Eout(G_t) as a function of iterations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ein) + 1), ein, label=r'$E_{in}(G_t)$', color='blue')
    plt.plot(range(1, len(eout) + 1), eout, label=r'$E_{out}(G_t)$', color='red')
    plt.xlabel('Iteration (t)')
    plt.ylabel('Error')
    plt.title(r'Generalization Error: $E_{in}(G_t)$ vs $E_{out}(G_t)$')
    plt.legend()
    plt.grid()
    plt.show()

# Paths to datasets
train_path = 'madelon'
test_path = 'madelon.t'
    
# Main script
if __name__ == "__main__":

    # Download datasets
    download_and_extract(train_data_url, train_path)
    download_and_extract(test_data_url, test_path)

    # Load data
    X_train, y_train = load_libsvm_data(train_path)
    X_test, y_test = load_libsvm_data(test_path)

    # Run AdaBoost
    T = 500
    ein_aggregate, eout_aggregate = adaboost_with_generalization(X_train, y_train, X_test, y_test, T)

    # Plot errors
    plot_generalization_error(ein_aggregate, eout_aggregate)