import sys
import tkinter as tk
from tkinter import messagebox
from typing import List

import torch
from loguru import logger

from core.config import AppConfig
from environments.base import DataFrame, StateType
from environments.connect4 import Connect4
from factories import get_agents, get_environment


class Connect4InspectorApp:
    def __init__(self, root, network, env: Connect4):
        self.root = root
        self.network = network
        self.env = env
        self.width = env.width
        self.height = env.height

        self.root.title("Connect4 Policy Inspector")

        self.board_state = [[None] * self.width for _ in range(self.height)]
        self.buttons: List[List[tk.Button]] = []

        self.player_colors = {0: "red", 1: "yellow", None: "lightgrey"}
        self.default_bg_hex = "#f0f0f0"

        # Create board UI
        board_frame = tk.Frame(root, padx=10, pady=10)
        board_frame.grid(row=0, column=0, columnspan=2)

        self.policy_labels: List[tk.Label] = []
        for c in range(self.width):
            lbl = tk.Label(board_frame, text=f"Col {c}\n-", width=8)
            lbl.grid(row=0, column=c)
            self.policy_labels.append(lbl)

        for row in range(self.height):
            row_buttons = []
            for col in range(self.width):
                button = tk.Button(
                    board_frame,
                    width=4,
                    height=2,
                    bg=self.player_colors[None],
                    command=lambda r=row, c=col: self.on_cell_click(r, c),
                )
                button.grid(row=row + 1, column=col, padx=2, pady=2)
                row_buttons.append(button)
            self.buttons.append(row_buttons)

        # Controls
        control_frame = tk.Frame(root, pady=10)
        control_frame.grid(row=1, column=0, sticky="ew", padx=10)

        tk.Label(control_frame, text="Current Player:").pack(side=tk.LEFT, padx=5)
        self.current_player_var = tk.IntVar(value=0)
        tk.Radiobutton(
            control_frame,
            text="Player 0",
            variable=self.current_player_var,
            value=0,
            command=self.predict,
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            control_frame,
            text="Player 1",
            variable=self.current_player_var,
            value=1,
            command=self.predict,
        ).pack(side=tk.LEFT)

        # Value display
        value_frame = tk.Frame(root, pady=10)
        value_frame.grid(row=1, column=1, sticky="ew", padx=10)
        tk.Label(value_frame, text="Predicted Value:").pack(side=tk.LEFT)
        self.value_label = tk.Label(value_frame, text="-", width=10)
        self.value_label.pack(side=tk.LEFT)

        self.predict()

    def _prob_to_color(self, prob: float) -> str:
        """Converts a probability (0-1) to a color for visualization."""
        if prob is None:
            return self.default_bg_hex
        # Interpolate between light grey and green
        start_rgb = (240, 240, 240)
        end_rgb = (144, 238, 144)  # LightGreen
        r = int(start_rgb[0] * (1 - prob) + end_rgb[0] * prob)
        g = int(start_rgb[1] * (1 - prob) + end_rgb[1] * prob)
        b = int(start_rgb[2] * (1 - prob) + end_rgb[2] * prob)
        return f"#{r:02x}{g:02x}{b:02x}"

    def on_cell_click(self, r, c):
        current_state = self.board_state[r][c]
        if current_state is None:
            new_state = 0
        elif current_state == 0:
            new_state = 1
        else:
            new_state = None

        self.board_state[r][c] = new_state
        self.buttons[r][c].config(bg=self.player_colors[new_state])
        self.predict()

    def predict(self):
        pieces_data = []
        for r in range(self.height):
            for c in range(self.width):
                if self.board_state[r][c] is not None:
                    pieces_data.append((r, c, self.board_state[r][c]))

        pieces_df = DataFrame(pieces_data, columns=["row", "col", "player_id"])

        winner = Connect4.check_for_winner_from_pieces(
            pieces_df, self.env.width, self.env.height
        )
        done = (winner is not None) or (
            pieces_df.height == self.env.width * self.env.height
        )
        current_player = self.current_player_var.get()

        game_df = DataFrame(
            data=[[current_player, done, winner]],
            columns=["current_player", "done", "winner"],
        )

        state: StateType = {"pieces": pieces_df, "game": game_df}

        temp_env = self.env.copy()
        temp_env.set_state(state)

        legal_actions = temp_env.get_legal_actions()

        # Reset policy labels
        for c in range(self.width):
            self.policy_labels[c].config(text=f"Col {c}\n-", bg=self.default_bg_hex)

        if not legal_actions:
            messagebox.showinfo("Prediction", "No legal actions in this state.")
            self.value_label.config(text="-")
            return

        with torch.no_grad():
            policy_dict, value = self.network.predict_single(
                temp_env.get_state_with_key(), legal_actions
            )

        self.value_label.config(text=f"{value:.4f}")

        for action, prob in policy_dict.items():
            if 0 <= action < self.width:
                color = self._prob_to_color(prob)
                self.policy_labels[action].config(
                    text=f"Col {action}\n{prob:.3f}", bg=color
                )


def main():
    """
    Main loop to inspect Connect4 board states with AlphaZero network.
    """
    config = AppConfig()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("--- Connect4 Policy Inspector ---")

    env = get_environment(config.env)
    if not isinstance(env, Connect4):
        logger.error("This tool is only for the Connect4 environment.")
        return

    agents = get_agents(env, config, load_all_az_iterations=False)
    agent = agents["AZ_400"]

    if not agent.load():
        logger.warning(
            "No checkpoint found for 'AlphaZero'. Using an untrained network."
        )
    else:
        logger.info("Loaded checkpoint for 'AlphaZero'")

    network = agent.network
    if not network:
        logger.error("Agent has no network.")
        return
    network.eval()

    root = tk.Tk()
    app = Connect4InspectorApp(root, network, env)
    root.mainloop()


if __name__ == "__main__":
    main()
