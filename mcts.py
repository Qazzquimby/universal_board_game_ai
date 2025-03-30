import math
import random
from typing import Dict, List, Tuple, Any


class MCTSNode:
    """A node in the Monte Carlo Tree Search"""

    def __init__(self, parent: "MCTSNode" = None, prior: float = 1.0):
        self.parent = parent
        self.prior = prior  # Prior probability from policy network
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[Any, MCTSNode] = {}

    @property
    def value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count else 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def expand(self, legal_actions: List[Tuple[int, int]]):
        """Create child nodes for all legal actions"""
        for action in legal_actions:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior=1.0)

    def select_child(self, exploration_constant: float) -> Tuple[Any, "MCTSNode"]:
        """Select child with highest UCB score"""
        _, action, node = max(
            (self._ucb_score(child, exploration_constant), action, child)
            for action, child in self.children.items()
        )
        return action, node

    def _ucb_score(self, child: "MCTSNode", c: float) -> float:
        """Calculate UCB1 score"""
        if child.visit_count == 0:
            q_value = 0.0
        else:
            q_value = child.value

        return q_value + c * (
            math.sqrt(math.log(self.visit_count + 1) / (child.visit_count + 1e-6))
        )


class MCTSAgent:
    """Monte Carlo Tree Search agent with random rollouts"""

    def __init__(
        self,
        env: "BoardGameEnv",
        num_simulations: int = 100,
        exploration_constant: float = 1.41,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        self.root = MCTSNode()
        # Cache for state keys to avoid recomputation
        self._state_cache = {}

    def _run_simulation(self, state: dict, root_env: "BoardGameEnv") -> float:
        """Run a single MCTS simulation"""
        node = self.root
        env = root_env.copy()
        depth = 0

        # Select - walk down tree until leaf node
        while node.is_expanded():
            action, node = node.select_child(self.exploration_constant)
            env.step(action)
            depth += 1

        # Expand and evaluate if game not ended
        if not env.is_game_over():
            legal_actions = env.get_legal_actions()
            node.expand(legal_actions)
            value = self._rollout(env) * (self.discount_factor**depth)
        else:
            if env.get_winning_player() == root_env.current_player:
                value = 1.0
            elif env.get_winning_player() is None:
                value = 0.0
            else:
                value = -1.0

        # Backpropagate
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent
            value = -value  # Alternate values for 2-player games

        return value

    def _rollout(self, env: "BoardGameEnv") -> float:
        """Random rollout policy"""
        current_player = env.get_current_player()
        while not env.is_game_over():
            action = random.choice(env.get_legal_actions())
            env.step(action)

        winner = env.get_winning_player()
        if winner is None:
            return 0.0
        return 1.0 if winner == current_player else -1.0

    def act(self, state: dict) -> Tuple[int, int]:
        """Choose action through MCTS search"""
        original_env = self.env
        for _ in range(self.num_simulations):
            self._run_simulation(state, original_env)

        # Choose most visited action
        best_action = max(
            self.root.children.items(), key=lambda item: item[1].visit_count
        )[0]

        # Move root to the selected child for next move
        if best_action in self.root.children:
            self.root = self.root.children[best_action]
            self.root.parent = None  # Allow old tree to be garbage collected
        else:
            self.root = MCTSNode()

        return best_action

    def reset(self) -> None:
        """Reset search tree between moves"""
        self.root = MCTSNode()
        self._state_cache.clear()
