'''
10. (20 points, human-graded) In class, we taught about the learning model of “positive and negative rays” (which is simply one-dimensional perceptron) for one-dimensional data. The model contains hypotheses of the form: h_(s,θ)(x) = s·sign(x-θ).
You can take sign(0)=-1 for simplicity but it should not matter much for the following problems. The model is frequently named the “decision stump” model and is one of the simplest learning models. As shown in class, for one-dimensional data, the growth function of the model is m_H (N)=2N and its VC dimension is 2.
In the following problems, you are asked to first play with decision stumps on an artificial data set. First, start by generating a one-dimensional data by the procedure below:
a) Generate x by a uniform distribution in [-1, 1].
b) Generate y by y = sign(x) + noise, where the noise flips the sign with probability 0 ≤ p < 1/2.
With the (x, y) generation process above, prove that for any h_(s,θ) with s ∈ {-1, +1} and θ ∈ [-1, 1],
E_out(h_(s,θ)) = u + v·|θ|, where v = s(1/2-p) and u = 1/2-v.

11. (20 points, code needed, human-graded) In fact, the decision stump model is one of the few models that we could minimize E_in efficiently by enumerating all possible thresholds. In particular, for N examples, there are at most 2N dichotomies (see the slides for positive rays), and thus at most 2N different E_in values. We can then easily choose the hypothesis that leads to the lowest E_in by the following decision stump learning algorithm.
(1) sort all N examples x_n to a sorted sequence x'_1 , x'_2 , ... , x'_N such that x'_1 ≤ x'_2 ≤ x'_3 ≤ ... ≤ x'_N
(2) for each θ ∈ {-1}U{ (x'_i + x'_(i+1))/2 : 1 ≤ i ≤ N -1 and x'_i != x'_i+1 } and s ∈ {-1, +1}, calculate E_in (h_(s,θ))
(3) return the h_(s,θ) with the minimum E_in as g; if multiple hypotheses reach the minimum E_in , return the one with the smallest s·θ.
(Hint: CS-majored students are encouraged to think about whether the second step can be carried out efficiently, i.e. O(N), using dynamic promgramming instead of the naive implementation of O(N^2).)

Generate a data set of size 12 by the procedure above with p = 15%, and run the one-dimensional decision stump algorithm on the data set to get g. Record E_in(g) and compute E_out(g) with the formula in Problem 10. Repeat the experiment 2000 times. Plot a scatter plot of (E_in(g), E_out(g)), and calculate the median of E_out(g) - E_in(g).
'''

import numpy as np
import matplotlib.pyplot as plt

# Experiment parameters
N = 12
p = 0.15
num_experiments = 2000
E_in_array = []
E_out_array = []

# sign function
def sign(x):
    return np.where(x > 0, 1, -1) # sign(0) = -1

# generate (x, y) dataset
def generate_data():
    x = np.random.uniform(-1, 1, N)
    y = sign(x)
    # Flip sign with probability p
    noise = np.random.rand(N) < p
    y[noise] = -y[noise]
    return x, y

# calculate E_in
def calculate_E_in(x, y, s, theta):
    h = s * sign(x - theta)
    return np.mean(h != y)

# Decision stump algorithm
def decision_stump(x, y):
    # (1) sort all N examples x_n to a sorted sequence x'_1 , x'_2 , ..., x'_N such that x'_1 ≤ x'_2 ≤ x'_3 ≤ ... ≤ x'_N
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # (2) for each θ ∈ {-1}U{ (x'_i + x'_(i+1))/2 : 1 ≤ i ≤ N -1 and x'_i != x'_i+1 } and s ∈ {-1, +1}, calculate E_in (h_(s,θ))
    thresholds = np.concatenate(([-1], (x_sorted[:-1] + x_sorted[1:]) / 2))
    best_s, best_theta, min_E_in = 0, 0, float('inf')
    
    # (3) return the h_(s,θ) with the minimum E_in as g; if multiple hypotheses reach the minimum E_in , return the one with the smallest s·θ.
    for theta in thresholds:
        for s in [-1, 1]:
            E_in = calculate_E_in(x_sorted, y_sorted, s, theta)
            if E_in < min_E_in: # smallest E_in
                min_E_in = E_in
                best_s = s
                best_theta = theta
            elif E_in == min_E_in:
                if s * theta < best_s * best_theta: # smallest s * theta
                    best_s = s
                    best_theta = theta

    return best_s, best_theta, min_E_in

# calculate E_out based on the formula in Problem 10
def calculate_E_out(s, theta):
    v = s * (0.5 - p)
    u = 0.5 - v
    E_out = u + v * abs(theta)
    return E_out

# Plot a scatter plot of (E_in(g), E_out(g))
def plot_scatter(E_in_array, E_out_array):
    plt.scatter(E_in_array, E_out_array, alpha=0.5)
    plt.xlabel('E_in(g)')
    plt.ylabel('E_out(g)')
    plt.title('Scatter plot of E_in(g) vs E_out(g)')
    plt.show()

def main():
    for _ in range(num_experiments):
        x, y = generate_data()
        s, theta, E_in = decision_stump(x, y)
        E_out = calculate_E_out(s, theta)
        E_in_array.append(E_in)
        E_out_array.append(E_out)
    
    # Calculate the median of E_out(g) - E_in(g)
    median_diff = np.median(np.array(E_out_array) - np.array(E_in_array))
    print(f"Median of E_out(g) - E_in(g): {median_diff}")
    
    # Scatter plot of (E_in(g), E_out(g))
    plot_scatter(E_in_array, E_out_array)

if __name__ == '__main__':
    main()
