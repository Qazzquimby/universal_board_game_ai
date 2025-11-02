from core.agent_interface import Agent
from core.config import AppConfig
from environments.base import BaseEnvironment
from environments.connect4.connect4 import Connect4
from factories import create_learning_agent


class HumanAgent(Agent):
    """An agent that gets actions from a human player via the console."""

    def act(self, env: BaseEnvironment, **kwargs):
        legal_actions = env.get_legal_actions()
        print(f"Legal actions: {legal_actions}")
        while True:
            try:
                action_str = input("Enter your action: ")
                action = int(action_str)
                if action in legal_actions:
                    return action
                else:
                    print("Invalid action. Please choose from legal actions.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def reset_game(self) -> None:
        pass


def play_game(env: BaseEnvironment, agent0: Agent, agent1: Agent):
    """Plays a single game, rendering the state for the human player."""
    env.reset()
    done = False

    for agent in (agent0, agent1):
        agent.reset_game()
        if hasattr(agent, "network") and agent.network:
            agent.network.eval()

    while not done:
        env.render()
        current_player_idx = env.get_current_player()
        current_agent = agent0 if current_player_idx == 0 else agent1

        if isinstance(current_agent, HumanAgent):
            print(f"\nYour turn (Player {current_player_idx}).")
            action = current_agent.act(env=env)
        else:
            print(f"\nAI's turn (Player {current_player_idx})...")
            action = current_agent.act(env=env)
            print(f"AI chose action: {action}")

        assert action is not None

        result = env.step(action)
        done = result.done

    print("\n--- Game Over ---")
    env.render()

    winner = env.get_winning_player()
    if winner is not None:
        print(f"Player {winner} wins!")
    else:
        print("It's a draw!")


def main():
    """Main function to run a human vs. AI game."""
    env = Connect4()

    ai_agent = create_learning_agent(
        model_type="alphazero",
        env=env,
        config=AppConfig(),
    )

    # --- Human Player Setup ---
    human_agent = HumanAgent()

    # --- Game Setup ---
    human_player_id = None
    while human_player_id not in [0, 1]:
        try:
            choice = input(
                "Do you want to play as Player 0 (first) or Player 1 (second)? [0/1]: "
            )
            human_player_id = int(choice)
        except (ValueError, IndexError):
            print("Invalid input. Please enter 0 or 1.")

    if human_player_id == 0:
        agent0, agent1 = human_agent, ai_agent
        print("\nYou are Player 0. You go first.")
    else:
        agent0, agent1 = ai_agent, human_agent
        print("\nYou are Player 1. The AI goes first.")

    play_game(env, agent0, agent1)


if __name__ == "__main__":
    main()
