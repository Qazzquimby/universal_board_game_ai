import abc
from typing import Any, Dict, List, Tuple, Optional, TypeVar

# Generic type for actions, can be Tuple[int, int], Tuple[int, int], int, etc.
ActionType = TypeVar('ActionType')
# Generic type for the state representation used by the agent (often a dict)
StateType = Dict[str, Any] # Keep as Dict for now, could be more generic later

class EnvInterface(abc.ABC):
    """Abstract base class for game environments."""

    @abc.abstractmethod
    def reset(self) -> StateType:
        """
        Reset the environment to its initial state.

        Returns:
            The initial state observation.
        """
        pass

    @abc.abstractmethod
    def step(self, action: ActionType) -> Tuple[StateType, float, bool]:
        """
        Take a step in the environment using the given action.

        Args:
            action: The action taken by the current player.

        Returns:
            A tuple containing:
            - next_state: The state observation after the action.
            - reward: The reward received by the player who just acted.
            - done: Boolean indicating if the game has ended.
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
    def copy(self) -> 'EnvInterface':
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
