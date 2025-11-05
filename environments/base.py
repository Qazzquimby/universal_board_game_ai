import abc
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, TypeVar, Union, Tuple, Any
import numpy as np

ActionType = TypeVar("ActionType")


class DataFrame:
    def __init__(self, data=None, columns=None, schema=None):
        # Internal properties:
        # self._data: Dict[str, np.ndarray] holding the raw, unmasked data columns.
        # self._mask: Optional[np.ndarray] of booleans, indicating which rows are visible.
        # self.columns: List[str] of visible column names.
        # self._col_to_idx: Dict mapping column name to index.
        #
        # A DataFrame can be a "view" on another DataFrame's data.
        # In such cases, _data is shared, and _mask determines the visible subset of rows.
        # Operations like filter() or slicing create views.
        # Operations like concat() or with_columns() materialize the view into a new DataFrame.

        self._mask = None  # No mask by default, all data is visible.

        if columns is not None:
            self.columns = columns
        elif schema is not None:
            self.columns = list(schema.keys())
        else:
            self.columns = []

        self._col_to_idx = {name: i for i, name in enumerate(self.columns)}
        self._data: Dict[str, np.ndarray] = {c: np.array([]) for c in self.columns}

        if data:
            if isinstance(data, dict):  # Already column-oriented
                self._data = {c: np.array(v) for c, v in data.items()}
                if not self.columns:
                    self.columns = list(data.keys())
            elif isinstance(data, list):
                if not data:
                    pass
                elif isinstance(data[0], dict):
                    if not self.columns:
                        self.columns = list(data[0].keys())
                    data_by_col = {c: [d.get(c) for d in data] for c in self.columns}
                    self._data = {c: np.array(v) for c, v in data_by_col.items()}
                elif isinstance(data[0], (list, tuple)):
                    if not self.columns:
                        raise ValueError("Columns must be provided for row data")
                    if data and len(data[0]) != len(self.columns):
                        raise ValueError("Row length must match number of columns")
                    data_T = list(zip(*data)) if data else [[] for _ in self.columns]
                    self._data = {
                        c: np.array(data_T[i]) for i, c in enumerate(self.columns)
                    }

        if not self.columns and self._data:
            self.columns = list(self._data.keys())
            self._col_to_idx = {name: i for i, name in enumerate(self.columns)}

    @property
    def height(self):
        if self._mask is not None:
            return np.sum(self._mask)
        if not self.columns or not self._data:
            return 0
        # This assumes all columns have the same length, which is enforced by constructor
        return len(self._data[self.columns[0]])

    def is_empty(self):
        return self.height == 0

    def hash(self):
        return sum(hash(row) for row in self.rows())

    def filter(self, conditions: Union[Tuple[str, Any], Dict[str, Any]]):
        if isinstance(conditions, tuple):
            conditions = {conditions[0]: conditions[1]}

        if self.is_empty():
            return DataFrame(columns=self.columns)

        # Create a new mask by applying conditions on top of the existing mask.
        if self._mask is not None:
            new_mask = self._mask.copy()
        else:
            base_height = (
                len(self._data[self.columns[0]]) if self.columns and self._data else 0
            )
            new_mask = np.ones(base_height, dtype=bool)

        for col, val in conditions.items():
            # Apply filter to the underlying data
            new_mask &= self._data[col] == val

        # Return a new DataFrame as a view
        new_df = DataFrame(columns=self.columns)
        new_df._data = self._data
        new_df._mask = new_mask
        return new_df

    def select(self, columns):
        # Materializes the selected columns of the current view into a new DataFrame.
        new_data = {col: self[col] for col in columns}
        return DataFrame(data=new_data, columns=columns)

    def rows(self):
        if self.is_empty():
            return []
        # Use __getitem__ to get masked column data
        column_data = [self[c] for c in self.columns]
        return [tuple(row) for row in np.array(column_data, dtype=object).T]

    def concat(self, other_df):
        if self.columns != other_df.columns:
            raise ValueError("DataFrames have different columns")

        if other_df.is_empty():
            return self.clone()
        if self.is_empty():
            return other_df.clone()

        # Materialize both dataframes' visible rows and concatenate them.
        self_data = {c: self[c] for c in self.columns}
        other_data = {c: other_df[c] for c in other_df.columns}

        new_data = {
            col: np.concatenate([self_data[col], other_data[col]])
            for col in self.columns
        }
        return DataFrame(data=new_data, columns=self.columns)

    def with_columns(self, updates_dict):
        for col_name in updates_dict:
            if col_name not in self.columns:
                raise ValueError(f"Column {col_name} not in DataFrame")

        if self.is_empty():
            new_row_dict = {c: [None] for c in self.columns}
            for col_name, value in updates_dict.items():
                new_row_dict[col_name] = (
                    [value] if not isinstance(value, list) else value
                )
            return DataFrame(data=new_row_dict, columns=self.columns)

        # This operation materializes the DataFrame view.
        df_to_update = self.clone()

        for col_name, value in updates_dict.items():
            if not isinstance(value, (list, np.ndarray)):
                value = np.full(df_to_update.height, value)
            else:
                value = np.array(value)

            if len(value) != df_to_update.height:
                raise ValueError("Length of update values must match DataFrame height")
            df_to_update._data[col_name] = value

        return df_to_update

    def __getitem__(self, key):
        if isinstance(key, str):
            if self._mask is not None:
                return self._data[key][self._mask]
            return self._data[key]

        if isinstance(key, (int, slice, np.ndarray)):  # Allow row indexing/slicing
            # This returns a new DataFrame as a view.
            new_df = DataFrame(columns=self.columns)
            new_df._data = self._data

            full_height = (
                len(self._data[self.columns[0]]) if self.columns and self._data else 0
            )
            if self._mask is not None:
                visible_indices = np.where(self._mask)[0]
                try:
                    sliced_indices = visible_indices[key]
                except IndexError:
                    # Slicing an empty view, result is empty.
                    sliced_indices = np.array([], dtype=int)
            else:
                # Slicing the full dataframe
                sliced_indices = np.arange(full_height)[key]

            new_mask = np.zeros(full_height, dtype=bool)
            # handle scalar index vs array of indices
            if np.isscalar(sliced_indices):
                if 0 <= sliced_indices < full_height:
                    new_mask[sliced_indices] = True
            else:
                new_mask[sliced_indices] = True
            new_df._mask = new_mask
            return new_df

        raise TypeError(f"Unsupported key type for DataFrame getitem: {type(key)}")

    def clone(self):
        # Materializes the current view into a new, independent DataFrame.
        new_data = {col: self[col].copy() for col in self.columns}
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
        hashed = hash(pickle.dumps(state))
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
        self._legal_actions: Optional[List[ActionType]] = None

    def reset(self) -> StateWithKey:
        """
        Reset the environment to its initial state.

        Returns:
            The initial state observation.
        """
        self._dirty = True
        self._legal_actions = None
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
        self._legal_actions = None
        return self._step(action)

    @abc.abstractmethod
    def _step(self, action: ActionType) -> ActionResult:
        pass

    def get_legal_actions(self) -> List[ActionType]:
        """
        Get a list of legal actions available in the current state.

        Returns:
            A list of valid actions.
        """
        # temp
        self._legal_actions = None

        if self._legal_actions is None:
            self._legal_actions = self._get_legal_actions()
        return self._legal_actions

    @abc.abstractmethod
    def _get_legal_actions(self) -> List[ActionType]:
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
