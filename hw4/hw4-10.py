'''
10. We use a real-world data set to study linear and polynomial regression. We will reuse the cpusmall scale data set at "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale" to save you some efforts. First, execute the following. The first four steps are similar to what you have done in Homework 3. The last step is new. We will take N = 64 (instead of 32). The data set contains 8192 examples. In each experiment, you are asked to

	(1) randomly sample N out of 8192 examples as your training data.
	(2) add x0 = 1 to each example, as always.
	(3) run linear regression on the N examples, using any reasonable implementations of X_dagger(hint-1) on the input matrix X, to get w_lin
	(4) evaluate Ein(w_lin) by averaging the squared error over the N examples; estimate Eout(w_lin) by averaging the squared error over the rest (8192-N) examples
	(5) implement the SGD algorithm for linear regression using the results on pages 10 and 12 of Lecture 11(hint-2). Pick one example uniformly at random in each iteration, take η = 0.01 and initialize w with w0 = 0. Run the algorithm for 100000 iterations, and record Ein(w_t) and Eout(w_t) whenever t is a multiple of 200.

Repeat the steps above 1126 times, each with a different random seed. Calculate the average (Ein(w_lin), Eout(w_lin)) over the 1126 experiments. Also, for every t = 200, 400, ... , calculate the average (Ein(w_t), Eout(w_t)) over the 1126 experiments. Plot the average Ein(w_t) as a function of t. On the same figure, plot the average Eout(w_t) as a function of t. Then, show the average Ein(w_lin) as a horizontal line, and the average Eout(w_lin) as another horizontal line.

hint-1: X_dagger = (X^T X)^(-1) X^T
hint-2: SGD logistic regression: w_t+1 = w_t + η * theta(-y_n * w_t^T x_n) y_n * x_n, and theta(-y_n * w_t^T x_n) y_n * x_n is the -gradient err(w_t, x_n, y_n) with respect to w_t.
'''

import numpy as np
import requests
import matplotlib.pyplot as plt

# Download and parse data
def download_and_parse_data():
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale'
    response = requests.get(url)
    data = response.text

    lines = data.strip().split('\n')
    labels = []
    features = []
    for line in lines:
        tokens = line.split()
        labels.append(float(tokens[0]))  # The first value is the target (y)
        feature = [1]  # x0 = 1
        for token in tokens[1:]:
            _, value = token.split(":")
            feature.append(float(value))
        features.append(feature)
    
    X = np.array(features)
    y = np.array(labels)

    return X, y

# Linear regression
def linear_regression(X, y):
    X_dagger = np.linalg.pinv(X) # pseudo-inverse of X
    wlin = X_dagger @ y # wline
    return wlin

# SGD implementation for linear regression with recording of Eout
def sgd_linear_regression(X_in, y_in, X_out, y_out, eta=0.01, iterations=100000, record_interval=200):
    w = np.zeros(X_in.shape[1])  # Initialize w with zeros
    Ein_t = []
    Eout_t = []
    
    for t in range(1, iterations + 1):
        i = np.random.randint(len(X_in))
        x_i = X_in[i]
        y_i = y_in[i]
        
        # Update rule (P. 12 of Leacture 11)
        gradient = -2 * (y_i - w @ x_i) * x_i
        w -= eta * gradient
        
        # Record Ein and Eout every 'record_interval' iterations
        if t % record_interval == 0:
            Ein_t.append(squared_error(y_in, X_in @ w))
            Eout_t.append(squared_error(y_out, X_out @ w))
    
    return w, Ein_t, Eout_t

# Squared error calculation
def squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Main experiment loop
def main():
    # Load data
    X, y = download_and_parse_data()

    # Parameters
    N = 64
    num_experiments = 1126
    record_interval = 200
    iterations = 100000

    Ein_lin_avg = []
    Eout_lin_avg = []
    Ein_sgd_avg = np.zeros(iterations // record_interval)
    Eout_sgd_avg = np.zeros(iterations // record_interval)
    
    for _ in range(num_experiments):
        # Sample training and testing sets
        indices = np.random.choice(len(X), N, replace=False)
        X_in, y_in = X[indices], y[indices]
        X_out, y_out = np.delete(X, indices, axis=0), np.delete(y, indices, axis=0)

        # Linear regression
        w_lin = linear_regression(X_in, y_in)
        Ein_lin, Eout_lin = squared_error(y_in, X_in @ w_lin), squared_error(y_out, X_out @ w_lin)
        
        Ein_lin_avg.append(Ein_lin)
        Eout_lin_avg.append(Eout_lin)

        # SGD with tracking Ein_t and Eout_t
        _, Ein_t, Eout_t = sgd_linear_regression(X_in, y_in, X_out, y_out, iterations=iterations, record_interval=record_interval)
        Ein_sgd_avg += np.array(Ein_t)
        Eout_sgd_avg += np.array(Eout_t)

    # Average results
    Ein_lin_avg = np.mean(Ein_lin_avg)
    Eout_lin_avg = np.mean(Eout_lin_avg)
    Ein_sgd_avg /= num_experiments
    Eout_sgd_avg /= num_experiments

    # Plotting
    plt.figure(figsize=(10, 6))
    t_values = np.arange(record_interval, iterations + 1, record_interval)
    
    # Plot Ein and Eout for SGD over time
    plt.plot(t_values, Ein_sgd_avg, label="Average Ein(w_t) for SGD")
    plt.plot(t_values, Eout_sgd_avg, label="Average Eout(w_t) for SGD")
    
    # Plot horizontal lines for Ein and Eout of linear regression
    plt.axhline(y=Ein_lin_avg, color='r', linestyle='--', label="Average Ein(w_lin) for Linear Regression")
    plt.axhline(y=Eout_lin_avg, color='b', linestyle='--', label="Average Eout(w_lin) for Linear Regression")
    
    # Labeling
    plt.xlabel("Iteration (t)")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Average Ein(w_t) and Eout(w_t) vs Iteration t (SGD) and Ein, Eout of Linear Regression")
    plt.show()

if __name__ == '__main__':
    main()
