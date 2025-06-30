from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Callable, Any, Literal, Type, Union, TypeVar


TriggerTiming = Literal["before", "after", "modify", "replace"]

EventT = TypeVar("EventT")


class Selector:
    """An object that represents a query for a target, not a specific target."""

    def __init__(
        self,
        description: str,
        filter_func: Callable[[Any, Any], bool],
        event_attr: str = "actor",
    ):
        self.description = description  # For debugging and logging
        self.filter_func = filter_func
        self.event_attr = event_attr

    def matches(self, owner: Any, event: Any) -> bool:
        """Checks if a given instance matches the selector's criteria."""
        target_obj = getattr(event, self.event_attr, None)
        return self.filter_func(owner, target_obj)


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


class GameEngine:
    def __init__(self):
        self.players = {}

        self.action_resolvers: dict[str, Callable[[Any], None]] = {}

        self.hooks: defaultdict[str, list[TriggeredAbility]] = defaultdict(list)

        self.event_stack = deque()
        self.game_log = []
        self.turn = 0
        self.is_processing = False

    # Not sure but this may want to reuse a more generic add entity. The toy game doesn't have other entities but most games do, eg cards.
    def add_player(self, player):
        self.players[player.name] = player

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
                self.run()

        action_caller.__name__ = name
        return action_caller

    def run(self):
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
            for hook in self._get_hooks_for_event("replace", action_name, event):
                print(f"  - Applying 'replace' hook from {hook.owner.name}'s ability")
                if hook.response(event) is not False:
                    print(f"    -> Event was REPLACED.")
                    replaced = True
                    break
            if replaced:
                continue

            # MODIFICATION PHASE
            for hook in self._get_hooks_for_event("modify", action_name, event):
                print(f"  - Applying 'modify' hook from {hook.owner.name}'s ability")
                hook.response(event)
                print(f"    -> Event modified to: {event}")

            # BEFORE PHASE
            for hook in self._get_hooks_for_event("before", action_name, event):
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
            for hook in self._get_hooks_for_event("after", action_name, event):
                print(f"  - Applying 'after' hook from {hook.owner.name}'s ability")
                hook.response(event)

            chain_depth -= 1

        self.is_processing = False

    def _get_hooks_for_event(
        self, trigger_type: str, action_name: str, event: Any
    ) -> list[TriggeredAbility]:
        """Finds hooks using the new flexible target filters."""
        matching_hooks = []

        for hook in self.hooks[action_name]:
            if hook.timing != trigger_type:
                continue

            owner = hook.owner
            match = None
            if isinstance(hook.filter, Selector):
                match = hook.filter.matches(owner, event)
            else:
                # Fallback to old behavior for non-selector filters
                actor = getattr(event, "actor", None)
                if isinstance(hook.filter, type) and actor:
                    match = isinstance(actor, hook.filter)
                elif actor:  # actor is an instance
                    match = actor is hook.filter

            if match:
                matching_hooks.append(hook)
        return matching_hooks


TargetFilter = Union[Selector, Type, Any]


class GameEntity:
    def __init__(self, engine: GameEngine, name: str):
        self.engine = engine
        self.name = name
        engine.players[self.name] = self

    def _add_hook(
        self,
        timing: TriggerTiming,
        target_filter: Union[Selector, Type, Any],
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
        self.engine.hooks[hook.on_event].append(hook)

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


class QuerySet:
    def __init__(self, results):
        self.results = results

    def where(self, **kwargs):
        filtered_results = []
        for entry in self.results:
            match = all(entry.params.get(k) == v for k, v in kwargs.items())
            if match:
                filtered_results.append(entry)
        return QuerySet(filtered_results)

    def __len__(self):
        return len(self.results)

    def __bool__(self):
        return len(self.results) > 0


# def get(action):
#     results = [log for log in engine.game_log if log.action_name == action.__name__]
#     return QuerySet(results)
