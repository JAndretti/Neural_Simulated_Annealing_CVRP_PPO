import numpy as np
import matplotlib.pyplot as plt


def p_accept(gain: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """
    Compute the Metropolis acceptance probability for a proposed move.

    Args:
        gain: Energy difference (current_cost - proposed_cost)
        temp: Current temperature

    Returns:
        Probability to accept the move, following exp(gain/temp) with clipping to [0,1]
    """
    return np.minimum(np.exp(gain / temp), np.ones_like(gain))


# Values of cost_improvement between -2 and 2
cost_improvement = np.linspace(-1, 1, 500)

# Temperatures to test
temperatures = [100, 1, 0.1, 0.01]

# Create the plot
plt.figure(figsize=(8, 6))

for temp in temperatures:
    acceptance_prob = p_accept(cost_improvement, temp)
    plt.plot(cost_improvement, acceptance_prob, label=f"Temperature = {temp}")

# Configure the plot
plt.title("Evolution of Acceptance Probability")
plt.xlabel("Cost Improvement (cost_improvement)")
plt.ylabel("Acceptance Probability")
plt.legend()
plt.grid(True)

# # Display the plot
# plt.show()
# Save the plot
plt.savefig("plots/acceptance_probability.png", dpi=300)
plt.close()
