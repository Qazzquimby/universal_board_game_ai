import abc
from typing import Any, Dict, List

from environments.base import BaseEnvironment


class Agent(abc.ABC):
    """Abstract base class for all agents."""

    @abc.abstractmethod
    def act(self, env: BaseEnvironment) -> Any:
        """
        Choose an action based on the current state.

        Returns:
            The action chosen by the agent.
        """
        pass

    # Removed episode_history from base class learn method.
    # Agents are responsible for their own learning mechanisms.
    # Implementations might learn from internal buffers or require specific data passed from a training loop.
    def learn(self) -> None:
        """
        Perform a learning step.
        Not all agents learn (e.g., RandomAgent). Default implementation does nothing.
        Subclasses should implement their specific learning logic (e.g., sampling from a buffer,
        processing a trajectory provided differently).
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

    def load(self) -> bool:
        """
        Load the agent's state.
        Default implementation raises NotImplementedError.

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
