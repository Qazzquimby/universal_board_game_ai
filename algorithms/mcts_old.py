import math
import random
from typing import Optional, Dict, Tuple
from loguru import logger
from environments.base import ActionType, BaseEnvironment


class MCTSNode_Old:
    def __init__(self, parent: Optional["MCTSNode_Old"] = None, prior: float = 1.0):
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[ActionType, MCTSNode_Old] = {}

    @property
    def value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count else 0.0

    def is_expanded(self) -> bool:
        return bool(self.children)

    def expand(self, action_priors: Dict[ActionType, float]):
        """
        Expand the node using prior probabilities from the network.
        Args:
            action_priors: A dictionary mapping legal actions to their prior probabilities.
        """
        children_added = 0
        for action, prior in action_priors.items():
            action_key = tuple(action) if isinstance(action, list) else action
            if action_key not in self.children:
                self.children[action_key] = MCTSNode_Old(parent=self, prior=prior)
                children_added += 1
            else:
                logger.warning(
                    f"  Node.expand: Child for action {action_key} already exists."
                    f" Was expand called multiple times?"
                )

        if action_priors:
            assert children_added > 0 or all(
                p == 0.0 for p in action_priors.values()
            ), f"Node expansion failed: non-zero priors provided but no children added. Priors: {action_priors}"
            assert self.children or all(
                p == 0.0 for p in action_priors.values()
            ), f"Node expansion failed: children dict empty despite non-zero priors. Priors: {action_priors}"


class MCTS_Old:
    """Core MCTS algorithm implementation (UCB1 + Random Rollout)."""

    def __init__(
        self,
        exploration_constant: float = 1.41,  # Standard UCB1 exploration constant
        discount_factor: float = 1.0,
        num_simulations: int = 100,
    ):
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        self.num_simulations = num_simulations
        self.root = MCTSNode_Old()

    def reset_root(self):
        """Resets the root node."""
        self.root = MCTSNode_Old()

    def _select(
        self, node: MCTSNode_Old, sim_env: BaseEnvironment
    ) -> Tuple[MCTSNode_Old, BaseEnvironment]:
        """Select child node with highest UCB score until a leaf node is reached."""
        while node.is_expanded() and not sim_env.state.done:
            parent_visits = node.visit_count

            child_scores = {
                act: self._score_child(node=child, parent_visits=parent_visits)
                for act, child in node.children.items()
            }
            if not child_scores:
                logger.warning("MCTS _select: Node expanded but no children found.")
                break

            best_action = max(child_scores, key=child_scores.get)
            action, node = best_action, node.children[best_action]

            sim_env.step(action)
        return node, sim_env

    def _score_child(self, node: MCTSNode_Old, parent_visits: int) -> float:
        """Calculate UCB1 score."""
        if node.visit_count == 0:
            return float("inf")
        exploration_term = self.exploration_constant * math.sqrt(
            math.log(max(1, parent_visits)) / node.visit_count
        )
        q_value_for_parent = -node.value
        return q_value_for_parent + exploration_term

    def _expand(self, node: MCTSNode_Old, env: BaseEnvironment):
        """Expand the leaf node by creating children for all legal actions."""
        if node.is_expanded() or env.state.done:
            return

        legal_actions = env.get_legal_actions()
        action_priors = {
            action: 1.0 / len(legal_actions) if legal_actions else 1.0
            for action in legal_actions
        }
        node.expand(action_priors)

    def _rollout(self, env: BaseEnvironment) -> float:
        """Simulate game from current state using random policy."""
        player_at_rollout_start = env.get_current_player()

        sim_env = env.copy()
        steps = 0
        max_steps = (
            env.width * env.height
            if hasattr(env, "width") and hasattr(env, "height")
            else 100
        )

        while not sim_env.state.done and steps < max_steps:
            legal_actions = sim_env.get_legal_actions()
            if not legal_actions:
                logger.warning("MCTS _rollout: Game not over, but no legal actions.")
                break
            action = random.choice(legal_actions)
            sim_env.step(action)
            steps += 1

        if steps >= max_steps:
            logger.warning(
                f"MCTS _rollout: Reached max steps ({max_steps}). Treating as draw."
            )
            return 0.0

        winner = sim_env.get_winning_player()
        value = sim_env.state.get_reward_for_player(player_at_rollout_start)
        return value

    def _backpropagate(
        self, leaf_node: MCTSNode_Old, value_from_leaf_perspective: float
    ):
        """
        Backpropagate the evaluated value up the tree, updating node statistics.

        Args:
            leaf_node: The node where the simulation/evaluation ended.
            value_from_leaf_perspective: The outcome (+1, -1, 0) from the perspective
                                         of the player whose turn it was at the leaf_node.
        """
        current_node = leaf_node
        value_for_current_node = value_from_leaf_perspective

        while current_node is not None:
            current_node.visit_count += 1
            current_node.total_value += value_for_current_node
            value_for_current_node *= -1
            current_node = current_node.parent

    def search(self, env: BaseEnvironment) -> MCTSNode_Old:
        """Run MCTS search from the given state using UCB1 and random rollouts."""
        self.reset_root()  # I think this prevents tree reuse
        state = env.get_state_with_key().state
        for sim_num in range(self.num_simulations):
            sim_env = env.copy()
            sim_env.set_state(state)

            leaf_node, leaf_env = self._select(self.root, sim_env)

            if not leaf_env.state.done:
                if not leaf_node.is_expanded():
                    self._expand(leaf_node, leaf_env)
                value = self._rollout(leaf_env)
            else:
                player_at_leaf = leaf_env.get_current_player()
                value = leaf_env.state.get_reward_for_player(player_at_leaf)

            self._backpropagate(leaf_node, value)

        return self.root
