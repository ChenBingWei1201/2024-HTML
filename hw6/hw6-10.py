import os
import requests
import bz2
from libsvm.svmutil import *

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

# Step 2: Load and preprocess the data
def load_data(file_path):
    y, x = svm_read_problem(file_path)
    labels, features = [], []
    for i in range(len(y)):
        if y[i] == 3 or y[i] == 7:
            labels.append(1 if y[i] == 3 else -1)  # Map class 3 to +1 and class 7 to -1
            features.append(x[i])
    return labels, features

labels, features = load_data(train_file)

# Step 3: Train SVM and count support vectors
def train_and_count_sv(labels, features, C, Q):
    param = f'-t 1 -d {Q} -c {C} -g 1 -r 1 -q' # -t 1: polynomial kernel
    model = svm_train(labels, features, param)
    return model.get_nr_sv()  # Return number of support vectors

C_values = [0.1, 1, 10]
Q_values = [2, 3, 4]
results = []

# Step 4: Test different combinations of C and Q
for C in C_values:
    for Q in Q_values:
        num_sv = train_and_count_sv(labels, features, C, Q)
        results.append((C, Q, num_sv))
        print(f"C={C}, Q={Q}, Support Vectors={num_sv}")

# Step 5: Output results in a table
print("\nResults:")
print(f"{'C':<5} {'Q':<5} {'#SV':<5}")
for C, Q, num_sv in results:
    print(f"{C:<5} {Q:<5} {num_sv:<5}")

# Find combination with the smallest number of support vectors
min_sv = min(results, key=lambda x: x[2])
print(f"\nCombination with the smallest number of support vectors: C={min_sv[0]}, Q={min_sv[1]}, #SV={min_sv[2]}")