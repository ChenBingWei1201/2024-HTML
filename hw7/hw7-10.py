import os
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import zero_one_loss

# Decision Stump Classifier
class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.sign = None

    def train(self, X, y, weights):
        _, n = X.shape
        best_error = float('inf')

        # Search for the best dimension, threshold, and sign
        for feature_index in range(n):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                for sign in [-1, 1]:
                    predictions = sign * np.sign(X[:, feature_index] - threshold)
                    error = np.sum(weights * (predictions != y))
                    if error < best_error:
                        best_error = error
                        self.feature_index = feature_index
                        self.threshold = threshold
                        self.sign = sign

        return best_error

    def predict(self, X):
        predictions = self.sign * np.sign(X[:, self.feature_index] - self.threshold)
        return predictions

# AdaBoost Algorithm
def adaBoost(X_train, y_train, X_test, y_test, T=500):
    m = X_train.shape[0]
    weights = np.ones(m) / m
    classifiers = []
    alphas = []

    # Metrics for plotting
    E_in = []
    epsilons = []

    for t in range(T):
        stump = DecisionStump()
        error = stump.train(X_train, y_train, weights)

        # Avoid divide-by-zero
        if error == 0:
            error = 1e-10

        # Compute alpha
        alpha = 0.5 * np.log((1 - error) / error)
        alphas.append(alpha)
        classifiers.append(stump)

        # Update weights
        predictions = stump.predict(X_train)
        weights *= np.exp(-alpha * y_train * predictions)

        # Compute 0/1 error (E_in) and epsilon
        E_in.append(zero_one_loss(y_train, np.sign(sum(alpha * clf.predict(X_train) for clf, alpha in zip(classifiers, alphas)))))
        epsilons.append(error)

        print(f"Iteration {t+1}: Error = {error:.4f}, Alpha = {alpha:.4f}, E_in = {E_in[-1]:.4f}")

    # Final testing error
    final_predictions = np.sign(sum(alpha * clf.predict(X_test) for clf, alpha in zip(classifiers, alphas)))
    E_out = zero_one_loss(y_test, final_predictions)

    print(f"Final Test Error (E_out): {E_out:.4f}")

    return E_in, epsilons


# Function to download and extract data
def download_and_extract(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {url} to {dest_path}...")
        response = requests.get(url)
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print("Download complete!")
    else:
        print(f"File already exists at {dest_path}.")

# Function to load LIBSVM formatted data
def load_data(file_path):
    data = load_svmlight_file(file_path)
    X, y = data[0].toarray(), data[1]
    y = np.where(y <= 0, -1, 1)  # Ensure labels are -1 or 1
    return X, y


# Main Function
if __name__ == "__main__":
    # URLs for datasets
    train_data_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/madelon'
    test_data_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/madelon.t'
    
    # Download datasets
    train_path = "madelon_train"
    test_path = "madelon_test"
    download_and_extract(train_data_url, train_path)
    download_and_extract(test_data_url, test_path)

    # Load data
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)

    # Run AdaBoost with Decision Stumps
    T = 500
    E_in, epsilons = adaBoost(X_train, y_train, X_test, y_test, T)

    # Plot E_in and epsilon
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T+1), E_in, label=r"$E_{in}(g_t)$ (0/1 error)", color="blue")
    plt.plot(range(1, T+1), epsilons, label=r"$\epsilon_t$ (normalized weighted error)", color="red")
    plt.xlabel("Iteration (t)")
    plt.ylabel("Error")
    plt.title("AdaBoost-Stump: $E_{in}$ and $\epsilon$ over Iterations")
    plt.legend()
    plt.grid()
    plt.show()