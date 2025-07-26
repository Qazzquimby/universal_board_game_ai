import abc
import pickle
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypeVar,
    Tuple,
    Callable,
    Literal,
    Union,
    Type,
    Generic,
    Iterable,
)

from pydantic import BaseModel, model_validator, ConfigDict, PrivateAttr, Field

ActionType = TypeVar("ActionType")
StateType = Dict[str, Any]


EventT = TypeVar("EventT")
TriggerTiming = Literal["before", "after", "modify", "replace"]


class Selector:
    """An object that represents a query for a target, not a specific target."""

    def __init__(
        self,
        description: str,
        filter_func: Callable[[Any, Any], bool],
        event_attr: str = "actor",
        event_props: dict = None,
    ):
        self.description = description  # For debugging and logging
        self.filter_func = filter_func
        self.event_attr = event_attr
        self.event_props = event_props or {}

    def matches(self, owner: Any, event: Any) -> bool:
        """Checks if a given instance matches the selector's criteria."""
        target_obj = getattr(event, self.event_attr, None)
        if not self.filter_func(owner, target_obj):
            return False

        for prop, value in self.event_props.items():
            if getattr(event, prop, None) != value:
                return False

        return True


@dataclass
class TriggeredAbility:
    timing: TriggerTiming
    filter: Union[Selector, Type, Any]
    on_event: str
    response: Callable
    owner: Any = None


@dataclass
class LogEntry:
    action_name: str
    event_data: Any
    turn: int


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
    reward: float = 0  #  The reward received by the player who just acted.
    done: bool = False


def mutator(method):
    def wrapper(self, *args, **kwargs):
        self._dirty = True
        return method(self, *args, **kwargs)

    return wrapper


PlayerId = int


class FeatureSpec(BaseModel):
    cardinality: int


NetworkableFeatures = Dict[str, FeatureSpec]


class Networkable(abc.ABC):
    @classmethod
    def get_feature_schema(cls, env: "BaseEnvironment") -> NetworkableFeatures:
        """
        Returns a schema for the entity's features for network configuration.
        Example: {"feature_name": FeatureSpec(cardinality=10)}
        """
        return {}

    def get_feature_values(self) -> Dict[str, int]:
        """
        Returns the concrete feature values for this entity instance.
        Example: {"feature_name": 3}
        """
        # todo take the dict of the subclass. Remove private attrs.
        raise NotImplementedError


TargetFilter = Union[Selector, Type, Any]


class GameEntity(BaseModel, Networkable, abc.ABC):
    name: str
    _env: Optional["BaseEnvironment"] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_feature_values(self) -> Dict[str, int]:
        """
        Returns the concrete feature values for this entity instance by dumping
        the model and filtering private attributes.
        Example: {"feature_name": 3}
        """
        all_attrs = self.model_dump()
        return {
            key: value for key, value in all_attrs.items() if not key.startswith("_")
        }

    def bind(self, env: "BaseEnvironment"):
        """Binds the entity to a live environment."""
        self._env = env
        env.entities[self.name] = self

    @property
    def env(self) -> "BaseEnvironment":
        if self._env is None:
            raise AttributeError(
                f"'{type(self).__name__}' instance '{self.name}' is not bound to an environment."
            )
        return self._env

    def _add_hook(
        self,
        timing: TriggerTiming,
        target_filter: TargetFilter,
        action: Callable,
        handler: Callable,
    ):
        hook = TriggeredAbility(
            timing=timing,
            filter=target_filter,
            on_event=action.__name__,
            response=handler,
            owner=self,
        )
        self.env.triggered_abilities[hook.on_event].append(hook)

    def modify(self, target_filter: TargetFilter, action: Callable, handler: Callable):
        self._add_hook(
            timing="modify",
            target_filter=target_filter,
            action=action,
            handler=handler,
        )

    def replace(self, target_filter: TargetFilter, action: Callable, handler: Callable):
        self._add_hook(
            timing="replace",
            target_filter=target_filter,
            action=action,
            handler=handler,
        )

    def before(self, target_filter: TargetFilter, action: Callable, handler: Callable):
        self._add_hook(
            timing="before",
            target_filter=target_filter,
            action=action,
            handler=handler,
        )

    def after(self, target_filter: TargetFilter, action: Callable, handler: Callable):
        self._add_hook(
            timing="after",
            target_filter=target_filter,
            action=action,
            handler=handler,
        )


class Player(GameEntity):
    id: PlayerId

    @classmethod
    def get_feature_schema(cls, env: "BaseEnvironment") -> NetworkableFeatures:
        # todo, automate.
        #  for attribute on this (or subclass), get its cardinality
        return {"id": FeatureSpec(cardinality=len(env.state.players) + 1)}


class Players(BaseModel, Iterable):
    players: List[Player] = Field(default_factory=list)
    current_index: int = 0

    # Constructed in post init
    num_players: Optional[int] = None
    player_labels: Optional[List[str]] = None

    def model_post_init(self, context: Any):
        if self.players:
            self.num_players = len(self.players)
            self.player_labels = [p.name for p in self.players]
        else:
            if self.player_labels is None:
                if self.num_players is None:
                    raise ValueError(
                        "Either num_players or player_labels must be provided"
                    )
                self.player_labels = [f"Player {i}" for i in range(self.num_players)]

            if self.num_players is None:
                self.num_players = len(self.player_labels)

            self.players = [
                Player(id=i, name=self.player_labels[i])
                for i in range(self.num_players)
            ]

        if len(self.player_labels) != self.num_players:
            raise ValueError("Number of player labels must match num_players.")
        assert self.players

    @property
    def current_player(self):
        return self.players[self.current_index]

    def set_to_next(self):
        self.current_index = (self.current_index + 1) % len(self.players)

    def set_to_previous(self):
        self.current_index = (self.current_index - 1) % len(self.players)

    def __len__(self):
        return len(self.players)

    def __iter__(self):
        for player in self.players:
            yield player

    def __getitem__(self, item):
        return self.players[item]


Cell_T = TypeVar("Cell_T")


class Grid(BaseModel, Generic[Cell_T]):
    width: int
    height: int

    cells: List[List[Optional[Cell_T]]] = None

    @model_validator(mode="after")
    def init_cells(self):
        if self.cells is None:
            self.cells = [[None for _ in range(self.width)] for _ in range(self.height)]
        return self

    def __getitem__(self, item: Tuple[int, int]) -> Optional[Cell_T]:
        return self.cells[item[0]][item[1]]

    def __setitem__(self, key: Tuple[int, int], value: Optional[Cell_T]) -> None:
        self.cells[key[0]][key[1]] = value

    def get_column(self, x: int) -> list[Optional[Cell_T]]:
        return [row[x] for row in self.cells]

    def get_row(self, y: int) -> list[Optional[Cell_T]]:
        return self.cells[y]

    def get_network_config(self, env: "BaseEnvironment") -> "NetworkConfig":
        """
        Derives the network configuration from this grid.
        It inspects the cell entity type to build the feature schema.
        """
        from typing import get_args, get_origin

        # Introspect the generic type of the grid's cells
        cells_annotation = self.model_fields["cells"].rebuild_annotation()
        # This is digging through List[List[Optional[Cell_T]]]
        list_of_optional_entity = get_args(cells_annotation)
        optional_entity = get_args(list_of_optional_entity[0])
        entity_type_arg = optional_entity[0]

        origin = get_origin(entity_type_arg)
        if origin is Union:
            # Handle Optional[Entity] -> get Entity
            non_none_args = [
                arg for arg in get_args(entity_type_arg) if arg is not type(None)
            ]
            entity_type = non_none_args[0] if non_none_args else None
        else:
            entity_type = entity_type_arg

        if not entity_type or not issubclass(entity_type, Networkable):
            raise TypeError("Grid cells must be a Networkable entity type.")

        features = entity_type.get_feature_schema(env)
        features["y"] = FeatureSpec(cardinality=self.height)
        features["x"] = FeatureSpec(cardinality=self.width)

        return NetworkConfig(features=features, entity_type=entity_type)

    def _is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def __str__(self) -> str:
        result = []
        for row in self.cells:
            result.append(
                "".join(str(cell.name) if cell is not None else "." for cell in row)
            )
        return "\n".join(result)

    def get_entities_with_position(
        self,
    ) -> Iterable[Tuple[Optional[Cell_T], Dict[str, int]]]:
        for row in range(self.height):
            for column in range(self.width):
                yield self.cells[row][column], {"y": row, "x": column}


class BaseState(BaseModel):
    players: Players

    rewards: dict[PlayerId, float] = {}
    done: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_reward_for_player(self, player_index):
        return self.rewards.get(player_index, 0)


class NetworkConfig(BaseModel):
    features: NetworkableFeatures
    entity_type: Type[Networkable]


class BaseEnvironment(abc.ABC):
    """Abstract base class for game environments."""

    def __init__(self):
        self._dirty = True
        self._state_with_key: Optional[StateWithKey] = None

        self.entities: dict[str, "GameEntity"] = {}
        self.action_resolvers: dict[str, Callable[[Any], None]] = {}
        self.triggered_abilities: defaultdict[
            str, list[TriggeredAbility]
        ] = defaultdict(list)
        self.event_stack = deque()
        self.game_log = []
        self.turn = 0
        self.is_processing = False

        self.state: Optional[BaseState] = None

    def _bind_state_entities(self):
        """
        Traverses the current state and binds any unbound GameEntity objects
        to this environment instance.
        """
        if not self.state:
            return

        # Bind players
        if hasattr(self.state, "players"):
            for player in self.state.players:
                if isinstance(player, GameEntity) and player._env is None:
                    player.bind(self)

        # Bind other entities, e.g., from a grid
        for entity, _position in self.get_all_networkable_entities():
            if isinstance(entity, GameEntity) and entity._env is None:
                entity.bind(self)

    def define_action(
        self,
        name: str,
        event_dataclass,
    ):
        """Defines a new action, its data structure, and its resolution logic."""
        self.action_resolvers[name] = event_dataclass.resolve

        def action_caller(*args, **kwargs) -> None:
            event = event_dataclass(*args, **kwargs)
            self.event_stack.append((name, event))
            if not self.is_processing:
                self.execute_event_stack()

        action_caller.__name__ = name
        return action_caller

    def execute_event_stack(self):
        self.is_processing = True
        chain_depth = 0
        MAX_DEPTH = 50

        while self.event_stack:
            if chain_depth > MAX_DEPTH:
                raise RuntimeError(
                    f"Event chain limit ({MAX_DEPTH}) exceeded. Potential infinite loop."
                )

            action_name, event = self.event_stack.popleft()
            print(f"\n--- Processing: {action_name}({event}) ---")
            chain_depth += 1

            # REPLACEMENT PHASE
            # A 'replace' hook cancels the original event by default.
            # The handler can return False to prevent this and let the event resolve.
            replaced = False
            for hook in self._get_triggered_abilities_for_event(
                "replace", action_name, event
            ):
                print(f"  - Applying 'replace' hook from {hook.owner.name}'s ability")
                if hook.response(event) is not False:
                    print(f"    -> Event was REPLACED.")
                    replaced = True
                    break
            if replaced:
                continue

            # MODIFICATION PHASE
            for hook in self._get_triggered_abilities_for_event(
                "modify", action_name, event
            ):
                print(f"  - Applying 'modify' hook from {hook.owner.name}'s ability")
                hook.response(event)
                print(f"    -> Event modified to: {event}")

            # BEFORE PHASE
            for hook in self._get_triggered_abilities_for_event(
                "before", action_name, event
            ):
                print(f"  - Applying 'before' hook from {hook.owner.name}'s ability")
                hook.response(event)

            # RESOLUTION PHASE
            print(f"  - Resolving: {action_name}({event})")
            resolver = self.action_resolvers.get(action_name)
            if resolver:
                resolver(event)
            else:
                print(f"    -> WARNING: No resolver found for action '{action_name}'")
            self.game_log.append(LogEntry(action_name, event, self.turn))

            # AFTER PHASE
            for hook in self._get_triggered_abilities_for_event(
                "after", action_name, event
            ):
                print(f"  - Applying 'after' hook from {hook.owner.name}'s ability")
                hook.response(event)

            chain_depth -= 1

        self.is_processing = False

    def _get_triggered_abilities_for_event(
        self, trigger_type: str, action_name: str, event: Any
    ) -> list[TriggeredAbility]:
        """Finds hooks using the new flexible target filters."""
        matching_hooks = []

        for triggered_ability in self.triggered_abilities[action_name]:
            if triggered_ability.timing != trigger_type:
                continue

            owner = triggered_ability.owner
            match = None
            if isinstance(triggered_ability.filter, Selector):
                match = triggered_ability.filter.matches(owner, event)
            else:
                # Fallback to old behavior for non-selector filters
                actor = getattr(event, "actor", None)
                if isinstance(triggered_ability.filter, type) and actor:
                    match = isinstance(actor, triggered_ability.filter)
                elif actor:  # actor is an instance
                    match = actor is triggered_ability.filter

            if match:
                matching_hooks.append(triggered_ability)
        return matching_hooks

    @abc.abstractmethod
    def map_action_to_policy_index(self, action: ActionType) -> Optional[int]:
        """Maps a specific action to its index in the policy output array."""
        pass

    @abc.abstractmethod
    def map_policy_index_to_action(self, index: int) -> Optional[ActionType]:
        """Maps a policy index back to a specific action."""
        pass

    @property
    @abc.abstractmethod
    def num_action_types(self) -> int:
        """The total number of distinct actions possible in the game."""
        pass

    def reset(self) -> StateWithKey:
        """
        Reset the environment to its initial state.

        Returns:
            The initial state observation.
        """
        self._dirty = True
        state_with_key = self._reset()
        self._bind_state_entities()
        return state_with_key

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

    def get_current_player(self) -> int:
        """
        Get the index of the player whose turn it is.

        Returns:
            The current player index (e.g., 0 or 1).
        """
        return self.state.players.current_index

    def get_network_config(self) -> NetworkConfig:
        """
        Provides the network configuration by inspecting the environment's state.

        The default implementation looks for a `Grid` instance in the state and
        delegates to its `get_network_config` method. Environments with
        different state structures should override this method.
        """
        # temp: find the first Grid in the state.
        grid_instance = None
        if self.state:
            for _key, value in self.state:
                if isinstance(value, Grid):
                    grid_instance = value
                    break

        if not grid_instance:
            raise NotImplementedError(
                "Default get_network_config requires a Grid in the state."
            )

        return grid_instance.get_network_config(self)

    def get_all_networkable_entities(
        self,
    ) -> Iterable[Tuple[Optional["Networkable"], Dict[str, int]]]:
        """
        Retrieves all networkable entities and their positions from the state.

        The default implementation looks for a `Grid` and returns its entities.
        Environments with different state structures should override this method.
        """
        # temp: find the first Grid in the state.
        grid_instance = None
        if self.state:
            for _key, value in self.state:
                if isinstance(value, Grid):
                    grid_instance = value
                    break

        if grid_instance:
            return grid_instance.get_entities_with_position()

        return []

    def get_state_with_key(self) -> StateWithKey:
        if self._dirty:
            self._state_with_key = StateWithKey.from_state(self._get_state())
            self._dirty = False
        return self._state_with_key

    @abc.abstractmethod
    def _get_state(self):
        pass

    def get_winning_player(self) -> Optional[int]:
        """
        Get the index of the winning player.

        Returns:
            The winner's index, or None if there is no winner (draw or game not over).
        """
        if self.state.done and self.state.rewards:
            return max(self.state.rewards, key=self.state.rewards.get)

        return None

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

        Implementations should call `self._bind_state_entities()` after setting
        the state to ensure all game entities are correctly bound.

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
