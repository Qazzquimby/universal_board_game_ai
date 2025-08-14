import abc
from dataclasses import dataclass
from typing import Dict, List, Optional, TypeVar

ActionType = TypeVar("ActionType")


class DataFrame:
    def __init__(self, data=None, columns=None, schema=None):
        if columns is not None:
            self.columns = columns
        elif schema is not None:
            self.columns = list(schema.keys())
        else:
            self.columns = []

        self._col_to_idx = {name: i for i, name in enumerate(self.columns)}
        self._data = []

        if data:
            if isinstance(data, list):
                if not data:  # empty list
                    pass
                elif isinstance(data[0], dict):
                    if not self.columns:  # infer columns from first dict
                        self.columns = list(data[0].keys())
                        self._col_to_idx = {
                            name: i for i, name in enumerate(self.columns)
                        }
                    for row_dict in data:
                        self._data.append([row_dict.get(c) for c in self.columns])
                elif isinstance(data[0], (list, tuple)):
                    self._data = [list(row) for row in data]

    @property
    def height(self):
        return len(self._data)

    def is_empty(self):
        return self.height == 0

    def hash(self):
        return sum(hash(tuple(row)) for row in self._data)

    def filter(self, condition):
        col_name, value = condition
        col_idx = self._col_to_idx[col_name]
        new_data = [row for row in self._data if row[col_idx] == value]
        return DataFrame(data=new_data, columns=self.columns)

    def select(self, columns):
        indices = [self._col_to_idx[col] for col in columns]
        new_data = [[row[i] for i in indices] for row in self._data]
        return DataFrame(data=new_data, columns=columns)

    def rows(self):
        return [tuple(row) for row in self._data]

    def concat(self, other_df):
        if self.columns != other_df.columns:
            raise ValueError("DataFrames have different columns")
        new_data = self._data + other_df._data
        return DataFrame(data=new_data, columns=self.columns)

    def with_columns(self, updates_dict):
        for col_name in updates_dict:
            if col_name not in self._col_to_idx:
                raise ValueError(f"Column {col_name} not in DataFrame")

        if self.is_empty():
            new_row_dict = {c: None for c in self.columns}
            new_row_dict.update(updates_dict)
            new_row = [new_row_dict[c] for c in self.columns]
            return DataFrame(data=[new_row], columns=self.columns)

        new_data = [list(row) for row in self._data]
        for col_name, value in updates_dict.items():
            col_idx = self._col_to_idx[col_name]
            if not isinstance(value, list):
                value = [value]
            for row, value_for_row in zip(new_data, value, strict=True):
                row[col_idx] = value_for_row

        return DataFrame(data=new_data, columns=self.columns)

    def __getitem__(self, key):
        if isinstance(key, str):
            col_idx = self._col_to_idx[key]
            return [row[col_idx] for row in self._data]
        raise TypeError(f"Unsupported key type for DataFrame getitem: {type(key)}")

    def clone(self):
        new_data = [list(row) for row in self._data]
        return DataFrame(data=new_data, columns=list(self.columns))


StateType = Dict[str, DataFrame]


@dataclass
class StateWithKey:
    state: StateType
    key: int

    @property
    def done(self) -> bool:
        """Checks if the game is over."""
        # This is specific to environments that have a 'game' table with a 'done' column.
        game_df = self.state.get("game")
        if game_df and not game_df.is_empty():
            return bool(game_df["done"][0])
        return False

    @classmethod
    def from_state(cls, state: StateType):
        key = cls._get_key_for_state(state)
        return cls(state=state, key=key)

    @staticmethod
    def _get_key_for_state(state: StateType) -> int:
        hashed = sum([v.hash() for v in state.values()])
        return hashed


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
    reward: float = 0.0  # The reward received by the player who just acted.
    done: bool = False


class BaseEnvironment(abc.ABC):
    """Abstract base class for game environments."""

    def __init__(self):
        self._dirty = True
        self._state_with_key: Optional[StateWithKey] = None
        self.state: Optional[StateType] = None

    def reset(self) -> StateWithKey:
        """
        Reset the environment to its initial state.

        Returns:
            The initial state observation.
        """
        self._dirty = True
        state_with_key = self._reset()
        self.state = state_with_key.state
        return state_with_key

    @property
    def is_done(self) -> bool:
        return self.state["game"]["done"][0]

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

    @abc.abstractmethod
    def get_current_player(self) -> int:
        """
        Get the index of the player whose turn it is.

        Returns:
            The current player index (e.g., 0 or 1).
        """
        pass

    def get_state_with_key(self) -> StateWithKey:
        if self._dirty:
            self._state_with_key = StateWithKey.from_state(self._get_state())
            self._dirty = False
        return self._state_with_key

    @abc.abstractmethod
    def _get_state(self) -> StateType:
        pass

    @abc.abstractmethod
    def get_winning_player(self) -> Optional[int]:
        """
        Get the index of the winning player.

        Returns:
            The winner's index, or None if there is no winner (draw or game not over).
        """
        pass

    @property
    def num_players(self) -> int:
        """The number of players in the game."""
        if not self.state or "players" not in self.state:
            raise RuntimeError(
                "Environment state not initialized or has no 'players' table. "
                "Cannot determine num_players."
            )
        return self.state["players"].height

    def get_reward_for_player(self, player=0) -> float:
        winner = self.get_winning_player()
        if winner is None:
            return 0.0  # Draw
        if winner == player:
            return 1.0
        else:
            return -1.0

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
        raise NotImplementedError

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

    def get_sanity_check_states(self) -> List[SanityCheckState]:
        """Returns a list of predefined states for sanity checking the environment."""
        return []

    @abc.abstractmethod
    def get_network_spec(self) -> Dict:
        """
        Returns a specification for the network architecture.
        This includes table schemas, feature cardinalities, and action space information.
        """
        pass
