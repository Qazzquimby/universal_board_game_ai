import matplotlib.pyplot as plt
import numpy as np

def plot_results(win_history, window_size=100):
    """Plot the training results (win/loss/draw rates)."""
    plt.figure(figsize=(12, 6))

    # Calculate win/draw/loss rates using a sliding window
    if len(win_history) >= window_size:
        win_rates = []
        draw_rates = []
        loss_rates = []

        # Ensure window doesn't exceed history length
        actual_window_size = min(window_size, len(win_history))

        for i in range(len(win_history) - actual_window_size + 1):
            window = win_history[i : i + actual_window_size]
            win_rates.append(window.count(1) / actual_window_size)
            draw_rates.append(window.count(0) / actual_window_size)
            loss_rates.append(window.count(-1) / actual_window_size)

        episodes = range(actual_window_size - 1, len(win_history))
        plt.plot(episodes, win_rates, "g-", label=f"Win Rate (Avg over {actual_window_size})")
        plt.plot(episodes, draw_rates, "y-", label=f"Draw Rate (Avg over {actual_window_size})")
        plt.plot(episodes, loss_rates, "r-", label=f"Loss Rate (Avg over {actual_window_size})")
        plt.legend()
        plt.ylim(-0.1, 1.1) # Set Y-axis limits for rates

    else:
        # Plot raw outcomes if not enough data for smoothing
        plt.plot(win_history, 'b.', label='Episode Outcome (1:Win, 0:Draw, -1:Loss)')
        plt.legend()
        plt.ylim(-1.1, 1.1) # Set Y-axis limits for outcomes


    plt.title("Agent Training Performance Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Rate / Outcome")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
