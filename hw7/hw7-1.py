N = 1126  # Total dataset size
threshold = 0.3  # Probability threshold for no duplicates

def calculate_no_duplicates(N, N_prime):
    prob = 1.0
    for i in range(N_prime):
        prob *= (N - i) / N
    return prob

# Check for each candidate N'
candidates = [40, 45, 50, 55, 60]
for N_prime in candidates:
    p_no_duplicates = calculate_no_duplicates(N, N_prime)
    p_at_least_one_duplicate = 1 - p_no_duplicates
    print(f"N' = {N_prime}, P(no duplicates) = {p_no_duplicates:.4f}, P(at least one duplicate) = {p_at_least_one_duplicate:.4f}")
    if p_no_duplicates < threshold:
        print(f"Minimum N' = {N_prime} satisfies the condition.")
        break