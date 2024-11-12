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

11. Next, consider the following homogeneous order-Q polynomial transform: Φ(x) = (1, x1 , x2 , ..., x12 , x1^2 , x2^2 , ..., x12^2 , ..., x1^Q , x2^Q , ..., x12^Q ). Transform the training and testing data according to Φ(x) with Q = 3, and again implement steps 1-4 in the previous problem on z_n = Φ(x_n)instead of x_n to get w_poly. Repeat the four steps above 1126 times, each with a different random seed. Plot a histogram of Ein^sqr(wlin) - Ein^sqr(w_poly).What is the average Ein^sqr(wlin) - Ein^sqr(w_poly)? This is the Ein gain for using (homogeneous) polynomial transform. 
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

# Polynomial transform
def polynomial_transform(X, Q=3):
    X_poly = [X]  # Start with the original features (degree 1)
    for q in range(2, Q + 1):
        X_poly.append(X[:, 1:] ** q)  # Skip x0 (already added in degree 1)
    return np.hstack(X_poly)

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

    Ein_lin_list = []
    Ein_poly_list = []
    
    for _ in range(num_experiments):
        # Sample training and testing sets
        indices = np.random.choice(len(X), N, replace=False)
        X_in, y_in = X[indices], y[indices]
        X_out, y_out = np.delete(X, indices, axis=0), np.delete(y, indices, axis=0)

        # Linear regression on original data
        w_lin = linear_regression(X_in, y_in)
        Ein_lin = squared_error(y_in, X_in @ w_lin)
        
        # Polynomial transformation for Q=3
        Z_in = polynomial_transform(X_in, Q=3)
        Z_out = polynomial_transform(X_out, Q=3)

        # Linear regression on transformed data
        w_poly = linear_regression(Z_in, y_in)
        Ein_poly = squared_error(y_in, Z_in @ w_poly)

        # Store Ein values for comparison
        Ein_lin_list.append(Ein_lin)
        Ein_poly_list.append(Ein_poly)

    # Calculate difference in Ein
    Ein_diff = np.array(Ein_lin_list) - np.array(Ein_poly_list)

    # Plot histogram of Ein differences
    plt.figure(figsize=(10, 6))
    plt.hist(Ein_diff, bins=30, edgecolor='black')
    plt.xlabel(r'$E_{\text{in}}^{\text{sqr}}(\mathbf{w}_{\text{lin}}) - E_{\text{in}}^{\text{sqr}}(\mathbf{w}_{\text{poly}})$')
    plt.ylabel("Number of Experiments")
    plt.title(r'Histogram of $E_{\text{in}}^{\text{sqr}}(\mathbf{w}_{\text{lin}}) - E_{\text{in}}^{\text{sqr}}(\mathbf{w}_{\text{poly}})$')
    plt.show()

    # Calculate and print average difference
    avg_Ein_diff = np.mean(Ein_diff)
    print(f'Average Ein(w_lin) - Ein(w_poly): {avg_Ein_diff}')

if __name__ == '__main__':
    main()
