import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypeVar, Tuple

ActionType = TypeVar("ActionType")
StateType = Dict[str, Any]


@dataclass
class SanityCheckState:
    """Holds data for a single sanity check case."""

    description: str
    state: StateType
    expected_value: Optional[float] = None
    expected_action: Optional[ActionType] = None


@dataclass
class ActionResult:
    next_state: StateType  # The state observation after the action.
    reward: float  #  The reward received by the player who just acted.
    done: bool


class BaseEnvironment(abc.ABC):
    """Abstract base class for game environments."""

    @property
    @abc.abstractmethod
    def observation_tensor_shape(self) -> Tuple[int, ...]:
        """Returns the shape of the flattened observation tensor for network input."""
        pass

    @property
    @abc.abstractmethod
    def policy_vector_size(self) -> int:
        """Returns the fixed size of the policy vector that networks should output."""
        pass

    @abc.abstractmethod
    def map_action_to_policy_index(self, action: ActionType) -> Optional[int]:
        """Maps an environment action to its corresponding policy vector index."""
        pass

    @abc.abstractmethod
    def map_policy_index_to_action(self, index: int) -> Optional[ActionType]:
        """Maps a policy vector index back to a valid environment action."""
        pass

    @abc.abstractmethod
    def reset(self) -> StateType:
        """
        Reset the environment to its initial state.

        Returns:
            The initial state observation.
        """
        pass

    @abc.abstractmethod
    def step(self, action: ActionType) -> ActionResult:
        """
        Take a step in the environment using the given action.

        Args:
            action: The action taken by the current player.
        """
        pass

    @abc.abstractmethod
    def get_legal_actions(self) -> List[ActionType]:
        """
        Get a list of legal actions available in the current state.

        Returns:
            A list of valid actions.
        """
        pass

    @property
    @abc.abstractmethod
    def num_players(self) -> int:
        """Number of players in the game."""
        pass

    @abc.abstractmethod
    def get_current_player(self) -> int:
        """
        Get the index of the player whose turn it is.

        Returns:
            The current player index (e.g., 0 or 1).
        """
        pass

    @abc.abstractmethod
    def is_game_over(self) -> bool:
        """
        Check if the game has ended.

        Returns:
            True if the game is over, False otherwise.
        """
        pass

    @abc.abstractmethod
    def get_observation(self) -> StateType:
        """
        Get the current state observation.

        Returns:
            The current state observation dictionary.
        """
        pass

    @abc.abstractmethod
    def get_winning_player(self) -> Optional[int]:
        """
        Get the index of the winning player.

        Returns:
            The winner's index, or None if there is no winner (draw or game not over).
        """
        pass

    @abc.abstractmethod
    def copy(self) -> "BaseEnvironment":
        """
        Create a deep copy of the environment state.

        Returns:
            A new instance of the environment with the same state.
        """
        pass

    @abc.abstractmethod
    def set_state(self, state: StateType) -> None:
        """
        Set the environment to a specific state.

        Args:
            state: The state dictionary to load.
        """
        pass

    @abc.abstractmethod
    def get_sanity_check_states(self) -> List[SanityCheckState]:
        """
        Get a list of predefined states for sanity checking agent predictions.

        Returns:
            A list of SanityCheckState objects.
        """
        pass

    def render(self, mode: str = "human") -> None:
        """
        Render the environment state (optional).

        Args:
            mode: The rendering mode (e.g., "human").
        """
        print("Rendering not implemented for this environment.")

    def close(self) -> None:
        """Clean up any resources (optional)."""
        pass
