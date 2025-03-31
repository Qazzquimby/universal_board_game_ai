# Standard library imports
import math
import random
from typing import List, Tuple, Any

# Local application imports
# Use the generic EnvInterface
from environments.env_interface import ActionType


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
