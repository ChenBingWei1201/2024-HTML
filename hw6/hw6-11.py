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

# Load the data
labels, features = svm_read_problem(train_file)

C_values = [0.1, 1, 10]
gamma_values = [0.1, 1, 10]

# Function to train the model and calculate the margin
def train_and_calculate_margin(labels, features, C, gamma):
    param = f'-t 2 -c {C} -g {gamma} -q'  # Gaussian kernel
    model = svm_train(labels, features, param)
    sv_coef = model.get_sv_coef()
    sv = model.get_SV()

    # Calculate ||w||: Sum the square of sv_coef values
    norm_w_squared = sum([sv_coef[i][0]**2 for i in range(len(sv_coef))])
    norm_w = norm_w_squared**0.5

    margin = 1 / norm_w
    return margin

# Train models and calculate margins
results = []
for C in C_values:
    for gamma in gamma_values:
        margin = train_and_calculate_margin(labels, features, C, gamma)
        results.append((C, gamma, margin))
        print(f"C={C}, gamma={gamma}, Margin={margin}")

# Output results in a table
print("\nResults:")
print(f"{'C':<5} {'gamma':<5} {'Margin':<10}")
for C, gamma, margin in results:
    print(f"{C:<5} {gamma:<5} {margin:<10.6f}")

# Find combination with the largest margin
max_margin = max(results, key=lambda x: x[2])
print(f"\nCombination with the largest margin: C={max_margin[0]}, gamma={max_margin[1]}, Margin={max_margin[2]}")