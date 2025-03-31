# Standard library imports
import math
import random
from typing import Dict, List, Tuple, Any, Optional

# Local application imports
# Use the generic EnvInterface
from core.env_interface import EnvInterface, StateType, ActionType
from core.agent_interface import Agent


class MCTSNode:
    """Core MCTS node that tracks search statistics"""

    def __init__(self, parent=None, prior=1.0):
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children = {}

    @property
    def value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count else 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    # Update action type hint
    def expand(self, legal_actions: List[ActionType]):
        """Create child nodes for all legal actions"""
        for action in legal_actions:
            # Ensure action is hashable if it's not already (e.g., list)
            action_key = tuple(action) if isinstance(action, list) else action
            if action_key not in self.children:
                self.children[action] = MCTSNode(parent=self)


class MCTS:
    """Core MCTS algorithm implementation, environment-agnostic"""

    def __init__(
        self, exploration_constant=1.41, discount_factor=1.0, num_simulations=100
    ):
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        self.num_simulations = num_simulations
        self.root = MCTSNode()

    def _ucb_score(self, node: MCTSNode, parent_visits: int) -> float:
        """Calculate UCB1 score"""
        if node.visit_count == 0:
            return float("inf")
        return node.value + self.exploration_constant * math.sqrt(
            math.log(parent_visits) / node.visit_count
        )

    def _select(self, node: MCTSNode, env) -> Tuple[MCTSNode, Any]:
        """Select child node with highest UCB score"""
        while node.is_expanded() and not env.is_game_over():
            action, node = max(
                (self._ucb_score(child, node.visit_count), action, child)
                for action, child in node.children.items()
            )[1:]
            # Action here is the key from children dict, should be hashable
            env.step(action)
        return node, env

    def _rollout(self, env) -> float:
        """Random rollout policy"""
        current_player = env.get_current_player()
        while not env.is_game_over():
            action = random.choice(env.get_legal_actions())
            env.step(action)

        winner = env.get_winning_player()
        if winner is None:
            return 0.0
        return 1.0 if winner == current_player else -1.0

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree"""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # Alternate for 2-player games
            node = node.parent

    def search(self, env, state) -> MCTSNode:
        """Run MCTS search from given state"""
        self.root = MCTSNode()
        for _ in range(self.num_simulations):
            sim_env = env.copy()
            sim_env.set_state(state)

            node, sim_env = self._select(self.root, sim_env)

            if not sim_env.is_game_over():
                legal_actions = sim_env.get_legal_actions()
                node.expand(legal_actions)
                value = self._rollout(sim_env)
            else:  # Terminal state found during selection
                winner = sim_env.get_winning_player()
                # Determine the player whose turn it *would* have been at this terminal node.
                # This is the perspective needed for the first step of backpropagation.
                player_at_terminal_node = sim_env.get_current_player()

                if winner is None:  # Draw
                    value = 0.0
                elif winner == player_at_terminal_node:  # Player whose turn it is wins
                    value = 1.0
                else:  # Player whose turn it is loses
                    value = -1.0
                # Note: This value is correct for the terminal node itself.
                # Backpropagation will flip the sign for the parent.

            self._backpropagate(node, value)

        return self.root


class MCTSAgent(Agent):
    """Agent that uses MCTS to choose actions."""

    def __init__(
        self,
        env: EnvInterface, # Use EnvInterface
        num_simulations: int = 100,
        exploration_constant: float = 1.41,
    ):
        """
        Args:
            env: The game environment instance (used for copy).
            num_simulations: Number of MCTS simulations per move.
            exploration_constant: Exploration constant (c) for UCB1.
        """
        self.env = env  # Keep env reference mainly for copy()
        # TODO: MuZero doesn't need an env?
        self.mcts = MCTS(
            num_simulations=num_simulations,
            exploration_constant=exploration_constant,
            discount_factor=1.0,  # Discount factor within the search tree
        )

    # Update action return type hint
    def act(self, state: StateType) -> ActionType:
        """
        Perform MCTS search and choose the best action.

        Args:
            state: The current environment state observation.

        Returns:
            The chosen action (row, col).
        """
        # Ensure the MCTS root is reset or updated appropriately.
        # For stateless MCTS between moves, we run a fresh search each time.
        # If we wanted to reuse the tree, we'd need more logic here.
        self.reset()  # Reset the tree for a fresh search from the current state

        root_node = self.mcts.search(self.env, state)

        if not root_node.children:
            # If root has no children after search (e.g., immediate terminal state or no legal moves)
            # Need a fallback strategy. Choosing randomly from legal moves is one option.
            print("Warning: MCTS root has no children after search. Falling back.")
            temp_env = self.env.copy()
            temp_env.set_state(state)
            legal_actions = temp_env.get_legal_actions()
            if legal_actions:
                return random.choice(legal_actions)
            else:
                # Return None if no action is possible, let caller handle it
                return None

        # Choose the action leading to the most visited child node
        best_action = max(
            root_node.children.items(), key=lambda item: item[1].visit_count
        )[0]
        return best_action

    def reset(self) -> None:
        """Reset the MCTS search tree."""
        self.mcts.root = MCTSNode()
