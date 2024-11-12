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
    
    # randomly generate some s ∈ {-1,+1} and θ ∈ [-1,1] uniformly
    theta = np.random.uniform(-1, 1)
    s = np.random.choice([-1, 1])
    E_in = calculate_E_in(x_sorted, y_sorted, s, theta)
    E_out = calculate_E_out(s, theta)

    return E_in, E_out

# calculate E_out based on the formula in Problem 10
def calculate_E_out(s, theta):
    v = s * (0.5 - p)
    u = 0.5 - v
    E_out = u + v * abs(theta)
    return E_out

# Plot a scatter plot of (E_in(g), E_out(g))
def plot_scatter(E_in_array, E_out_array):
    plt.scatter(E_in_array, E_out_array, alpha=0.5)
    plt.xlabel('E_in(g_RND)')
    plt.ylabel('E_out(g_RND)')
    plt.title('Scatter plot of E_in(g_RND) vs E_out(g_RND)')
    plt.show()


def main():
    for _ in range(num_experiments):
        x, y = generate_data()
        E_in, E_out = decision_stump(x, y)
        E_in_array.append(E_in)
        E_out_array.append(E_out)
    
    median_diff = np.median(np.array(E_out_array) - np.array(E_in_array))
    print(f"Median of E_out(g_RND) - E_in(g_RND): {median_diff}")
    plot_scatter(E_in_array, E_out_array)

if __name__ == '__main__':
    main()
