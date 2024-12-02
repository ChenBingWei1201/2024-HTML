import os
import requests
import bz2
from libsvm.svmutil import *
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from collections import Counter

train_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2"
train_file_compressed = "mnist.scale.bz2"
train_file = "mnist.scale"

# Step 1: Download and decompress the data
def download_and_extract(url, dest_path):
    if not os.path.exists(dest_path):
        response = requests.get(url)
        with open(dest_path, 'wb') as f:
            f.write(response.content)
    if dest_path.endswith('.bz2'):
        with bz2.BZ2File(dest_path) as fr, open(dest_path[:-4], 'wb') as fw:
            fw.write(fr.read())

download_and_extract(train_url, train_file_compressed)

def load_data():
    y, x = svm_read_problem(train_file)
    return np.array(y), np.array(x)

# Train and evaluate the SVM model
def train_and_evaluate(y_train, x_train, y_val, x_val, C, gamma):
    model = svm_train(y_train, x_train, f'-t 2 -c {C} -g {gamma} -q')
    _, p_acc, _ = svm_predict(y_val, x_val, model, '-q')
    return 100 - p_acc[0]

y, x = load_data()
C = 1
gamma_values = [0.01, 0.1, 1, 10, 100]
gamma_selection_count = Counter()

for _ in range(128):
    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=200, stratify=y)

    errors = []
    for gamma in gamma_values:
        error = train_and_evaluate(y_train, x_train, y_val, x_val, C, gamma)
        errors.append((gamma, error))

    # Select the gamma with the smallest error, break ties by choosing the smallest gamma
    best_gamma = min(errors, key=lambda item: (item[1], item[0]))[0]
    gamma_selection_count[best_gamma] += 1


# Plot the results
log_gamma_values = [math.log10(gamma) for gamma in gamma_selection_count.keys()]
plt.bar(log_gamma_values, gamma_selection_count.values())
plt.xlabel('log10(Gamma)')
plt.ylabel('Selection Frequency')
plt.title('log10(Gamma) vs Selection Frequency')
plt.show()
