import math
import random
from typing import Optional, Dict, Tuple
from loguru import logger
from environments.base import ActionType, BaseEnvironment, StateType


class MCTSNode_Old:
    def __init__(self, parent: Optional["MCTSNode_Old"] = None, prior: float = 1.0):
        self.parent = parent
        self.prior = prior  # P(s,a) - Prior probability from the network
        self.visit_count = 0
        self.total_value = 0.0  # W(s,a) - Total action value accumulated
        self.children: Dict[ActionType, MCTSNode_Old] = {}

    @property
    def value(self) -> float:  # Q(s,a) - Mean action value
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
                # logger.trace(f"  Node.expand: Creating child for action {action_key} with prior {prior:.4f}")
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
        discount_factor: float = 1.0,  # Discount factor for rollout rewards
        num_simulations: int = 100,
    ):
        self.exploration_constant = exploration_constant
        self.discount_factor = (
            discount_factor  # Not used in standard UCB1 backprop here
        )
        self.num_simulations = num_simulations
        self.root = MCTSNode_Old()

    def reset_root(self):
        """Resets the root node."""
        self.root = MCTSNode_Old()

    def _select(
        self, node: MCTSNode_Old, sim_env: BaseEnvironment
    ) -> Tuple[MCTSNode_Old, BaseEnvironment]:
        """Select child node with highest UCB score until a leaf node is reached."""
        while node.is_expanded() and not sim_env.is_game_over():
            parent_visits = node.visit_count

            # Select the action corresponding to the child with the highest UCB score
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
        if node.is_expanded() or env.is_game_over():
            return

        legal_actions = env.get_legal_actions()
        # Use uniform prior for standard MCTS expansion
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
        )  # Safety break

        while not sim_env.is_game_over() and steps < max_steps:
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
        if winner is None:
            value = 0.0
        elif winner == player_at_rollout_start:
            value = 1.0
        else:
            value = -1.0
        # logger.debug(f"  Rollout: StartPlayer={player_at_rollout_start}, Winner={winner}, Value={value}")
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
            # logger.debug(f"  Backprop: Node={current_node}, ValueToAdd={value_for_current_node:.3f}, OldW={current_node.total_value:.3f}, OldN={current_node.visit_count - 1}, NewW={current_node.total_value + value_for_current_node:.3f}, NewN={current_node.visit_count}")
            current_node.total_value += value_for_current_node

            # Flip the value perspective for the parent node.
            value_for_current_node *= -1
            current_node = current_node.parent

    def search(self, env: BaseEnvironment, state: StateType) -> MCTSNode_Old:
        """Run MCTS search from the given state using UCB1 and random rollouts."""
        self.reset_root()

        for sim_num in range(self.num_simulations):
            sim_env = env.copy()
            sim_env.set_state(state)

            # 1. Selection: Find a leaf node using UCB1.
            leaf_node, leaf_env_state = self._select(self.root, sim_env)

            # 2. Evaluation: Get the value of the leaf node.
            if not leaf_env_state.is_game_over():
                if not leaf_node.is_expanded():
                    self._expand(leaf_node, leaf_env_state)
                value = self._rollout(leaf_env_state)

            else:
                # Game ended during selection. Determine the outcome.
                # Value must be from the perspective of the player whose turn it was AT THE LEAF node.
                player_at_leaf = leaf_env_state.get_current_player()
                winner = leaf_env_state.get_winning_player()

                if winner is None:
                    value = 0.0
                elif winner == player_at_leaf:
                    value = 1.0
                else:
                    value = -1.0

                # logger.debug(f"  Terminal Found during Select: Winner={winner}, PlayerAtNode={player_at_leaf}, Value={value}")
            # 3. Backpropagation: Update nodes along the path from the leaf to the root.
            self._backpropagate(leaf_node, value)

        return self.root
