import abc
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypeVar, Tuple

ActionType = TypeVar("ActionType")
StateType = Dict[str, Any]


@dataclass
class StateWithKey:
    state: StateType
    key: int

    @classmethod
    def from_state(cls, state):
        key = cls._get_key_for_state(state)
        return cls(state=state, key=key)

    @staticmethod
    def _get_key_for_state(state):
        serialized = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        return hash(serialized)


@dataclass
class SanityCheckState:
    """Holds data for a single sanity check case."""

    description: str
    state_with_key: StateWithKey
    expected_value: Optional[float] = None
    expected_action: Optional[ActionType] = None


@dataclass
class ActionResult:
    next_state_with_key: StateWithKey
    reward: float  #  The reward received by the player who just acted.
    done: bool


def mutator(method):
    def wrapper(self, *args, **kwargs):
        self._dirty = True
        return method(self, *args, **kwargs)

    return wrapper


class BaseEnvironment(abc.ABC):
    """Abstract base class for game environments."""

    def __init__(self):
        self._dirty = True
        self._state_with_key: Optional[StateWithKey] = None

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

    def reset(self) -> StateWithKey:
        """
        Reset the environment to its initial state.

        Returns:
            The initial state observation.
        """
        self._dirty = True
        return self._reset()

    @abc.abstractmethod
    def _reset(self) -> StateWithKey:
        pass

    def step(self, action: ActionType) -> ActionResult:
        """
        Take a step in the environment using the given action.

        Args:
            action: The action taken by the current player.
        """
        self._dirty = True
        return self._step(action)

    @abc.abstractmethod
    def _step(self, action: ActionType) -> ActionResult:
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

    def get_state_with_key(self) -> StateWithKey:
        if self._dirty:
            self._state_with_key = StateWithKey.from_state(self._get_state())
        else:
            return self._state_with_key

    @abc.abstractmethod
    def _get_state(self):
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
