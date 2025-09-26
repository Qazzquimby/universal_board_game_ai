import matplotlib.pyplot as plt
from typing import List


def plot_losses(
    total_losses: List[float], value_losses: List[float], policy_losses: List[float]
):
    # should just use wandb
    """Plots the training losses over iterations."""
    if not total_losses:  # Check if any losses were recorded
        print("No loss data recorded to plot.")
        return

    iterations = range(1, len(total_losses) + 1)

    plt.figure(figsize=(12, 8))

    # Plot Total Loss
    plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st subplot
    plt.plot(iterations, total_losses, "b-", label="Total Loss")
    plt.title("Training Losses Over Iterations")
    plt.ylabel("Total Loss")
    plt.grid(True)
    plt.legend()

    # Plot Value Loss
    plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd subplot
    plt.plot(iterations, value_losses, "r-", label="Value Loss")
    plt.ylabel("Value Loss")
    plt.grid(True)
    plt.legend()

    # Plot Policy Loss
    plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd subplot
    plt.plot(iterations, policy_losses, "g-", label="Policy Loss")
    plt.xlabel("Training Iteration (Learn Step)")
    plt.ylabel("Policy Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
