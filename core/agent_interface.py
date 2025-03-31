import abc
from typing import Any, Dict, List


class Agent(abc.ABC):
    """Abstract base class for all agents."""

    @abc.abstractmethod
    def act(self, state: Dict[str, Any]) -> Any:
        """
        Choose an action based on the current state.

        Args:
            state: The current environment state observation.

        Returns:
            The action chosen by the agent.
        """
        pass

    def learn(self, episode_history: List[tuple]) -> None:
        """
        Update the agent's internal state based on experience from an episode.
        Not all agents learn (e.g., RandomAgent). Default implementation does nothing.

        Args:
            episode_history: A list of (state, action, reward, done) tuples from an episode.
        """
        pass  # Optional method

    def save(self) -> None:
        """
        Save the agent's state (e.g., model weights, Q-table).
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"Save method not implemented for {self.__class__.__name__}"
        )

    def load(self, filepath: str) -> bool:
        """
        Load the agent's state.
        Default implementation raises NotImplementedError.

        Args:
            filepath: The path to load the agent state from.

        Returns:
            True if loading was successful, False otherwise.
        """
        raise NotImplementedError(
            f"Load method not implemented for {self.__class__.__name__}"
        )

    def reset(self) -> None:
        """
        Reset any internal state of the agent (e.g., MCTS tree).
        Default implementation does nothing.
        """
        pass  # Optional method
