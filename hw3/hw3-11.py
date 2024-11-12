'''
10. We use a real-world data set to study linear regression. Please download the cpusmall scale data set at 
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale 
Example of data set: `90 1:-0.993496 2:-0.993043 3:-0.850291 4:-0.963479 5:-0.960727 6:-0.900596 7:-0.96642 8:-0.863996 9:-0.606175 10:-0.999291 11:0.0811894 12:0.651101`
We use the column-scaled version instead of the original version so you'd likely encounter fewer numerical issues. The data set contains 8192 examples. In each experiment, you are asked to
(1) randomly sample N out of 8192 examples as your training data.
(2) add x0 = 1 to each example, as always.
(3) run linear regression on the N examples, using any reasonable implementations of X_dagger on the input matrix X, to get wlin (hint: X_dagger = (X^T X)^(-1) X^T)
(4) evaluate Ein(wlin) by averaging the squared error over the N examples; estimate Eout(wlin) by averaging the squared error over the rest (8192-N) examples

For N = 32, run the experiment above for 1126 times, and plot a scatter plot of (Ein(wlin), Eout(wlin)) in each experiment.

11. For each of N = 25,50,75,100, ... ,2000, calculate Ein(N) and Eout(N) by averaging Ein and Eout over 16 experiments. Then, plot the learning curves that show Ein(N) and Eout(N) as functions of N on the same figure.
'''

import numpy as np
import requests
import matplotlib.pyplot as plt

def download_and_parse_data():
	# Downloading the dataset
	url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale'
	response = requests.get(url)
	data = response.text

	# Preprocess the data
	lines = data.strip().split('\n')
	labels = []
	features = []
	for line in lines:
		tokens = line.split()
		labels.append(int(tokens[0]))  # The first value is the target (y)
		feature = [1]  # x0 = 1
		for token in tokens[1:]:
			_, value = token.split(":")
			feature.append(float(value))
		features.append(feature)
	
	X = np.array(features)
	y = np.array(labels)

	return X, y

# Linear regression implementation
def linear_regression(X, y):
    X_dagger = np.linalg.pinv(X)  # Compute pseudo-inverse of X
    wlin = X_dagger @ y  # Compute wlin
    return wlin

# Squared error calculation
def squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Error calculation for training and test sets
def calculate_errors(X_in, y_in, X_out, y_out, wlin):
    y_in_pred = X_in @ wlin
    y_out_pred = X_out @ wlin
    Ein = squared_error(y_in, y_in_pred)
    Eout = squared_error(y_out, y_out_pred)
    return Ein, Eout

# Plot the learning curves
def plot_learning_curves(N_array, Ein_avg, Eout_avg):
	plt.plot(N_array, Ein_avg, label='Ein bar(N)', marker='o')
	plt.plot(N_array, Eout_avg, label='Eout bar(N)', marker='o')
	plt.xlabel('N')
	plt.ylabel('Error')
	plt.title('Learning Curves: Ein bar(N) and Eout bar(N)')
	plt.legend()
	plt.show()

# Experiment parameters
num_examples = 8192
N_array = range(25, 2001, 25)  # N values from 25 to 2000
num_experiments = 16

def main():
	# Download and parse the data
	X, y = download_and_parse_data()
	Ein_avg = []
	Eout_avg = []
	for N in N_array:
		Ein_array = []
		Eout_array = []

		# Run the experiment for the current N
		for _ in range(num_experiments):
			indices = np.random.choice(num_examples, N, replace=False)  # Randomly sample N examples for training
			X_in = X[indices]
			y_in = y[indices]
			mask = np.ones(num_examples, dtype=bool)
			mask[indices] = False
			X_out = X[mask]
			y_out = y[mask]

			# Run linear regression
			wlin = linear_regression(X_in, y_in)

			# Calculate Ein and Eout
			Ein, Eout = calculate_errors(X_in, y_in, X_out, y_out, wlin)

			# Store the results
			Ein_array.append(Ein)
			Eout_array.append(Eout)

		# Calculate average Ein and Eout for the current N
		Ein_avg.append(np.mean(Ein_array))
		Eout_avg.append(np.mean(Eout_array))

	# Plot the learning curves
	plot_learning_curves(N_array, Ein_avg, Eout_avg)

if __name__ == '__main__':
    main()