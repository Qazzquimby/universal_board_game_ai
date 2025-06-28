from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class Hook:
    hook_type: str
    target_filter: Callable
    action_name: str
    handler: Callable
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

        self.hooks: defaultdict[str, list[Hook]] = defaultdict(list)

        self.event_stack = deque()
        self.game_log = []
        self.turn = 0
        self.is_processing = False

    # Not sure but this may want to reuse a more generic add entity. The toy game doesn't have other entities but most games do, eg cards.
    def add_player(self, player):
        self.players[player.name] = player
        player.game = self

    def define_action(self, name: str, event_dataclass: type, resolver: Callable):
        """Defines a new action, its data structure, and its resolution logic."""
        self.action_resolvers[name] = resolver

        def action_caller(**kwargs):
            event = event_dataclass(**kwargs)
            self.event_stack.append((name, event))
            if not self.is_processing:
                self.run()

        action_caller.__name__ = name

        return action_caller

    def modify(
        self, owner: Any, target_filter: Callable, action: Callable, handler: Callable
    ):
        hook = Hook("modify", target_filter, action.__name__, handler, owner)
        self.hooks[hook.action_name].append(hook)

    def replace(
        self, owner: Any, target_filter: Callable, action: Callable, handler: Callable
    ):
        hook = Hook("replace", target_filter, action.__name__, handler, owner)
        self.hooks[hook.action_name].append(hook)

    def before(
        self, owner: Any, target_filter: Callable, action: Callable, handler: Callable
    ):
        hook = Hook("before", target_filter, action.__name__, handler, owner)
        self.hooks[hook.action_name].append(hook)

    def after(
        self, owner: Any, target_filter: Callable, action: Callable, handler: Callable
    ):
        hook = Hook("after", target_filter, action.__name__, handler, owner)
        self.hooks[hook.action_name].append(hook)

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
            # todo seems messy
            # Cases:
            # - There is no replacement, nothing happens. Event may require conditions that aren't met
            # - Handler triggers one or more events and this event is replaced
            # - Event is replaced by nothing. Could be called prevent.
            # There's probably a difference between "gets cancelled by some reaction" and "is made an illegal move" and the illegal moves should be understood by get_legal_moves
            # Note that events shouldn't be replaced with the same event type, that's modify
            # Maybe should return True, or call evt.cancel() or something to cancel the original
            replaced = False
            for hook in self._get_hooks_for_event("replace", action_name, event):
                print(f"  - Applying 'replace' hook from {hook.owner.name}'s ability")
                start_len = len(self.event_stack)
                hook.handler(event)
                if len(self.event_stack) > start_len:
                    print(f"    -> Event was REPLACED.")
                    replaced = True
                    break
            if replaced:
                continue

            # MODIFICATION PHASE
            for hook in self._get_hooks_for_event("modify", action_name, event):
                print(f"  - Applying 'modify' hook from {hook.owner.name}'s ability")
                hook.handler(event)
                print(f"    -> Event modified to: {event}")

            # BEFORE PHASE
            for hook in self._get_hooks_for_event("before", action_name, event):
                print(f"  - Applying 'before' hook from {hook.owner.name}'s ability")
                hook.handler(event)

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
                hook.handler(event)

            chain_depth -= 1

        self.is_processing = False

    def _get_hooks_for_event(
        self, hook_type: str, action_name: str, event: Any
    ) -> list[Hook]:
        """Finds hooks using the new flexible target filters."""
        matching_hooks = []
        for h in self.hooks[action_name]:
            if h.hook_type == hook_type:
                if h.target_filter(h.owner, event):
                    matching_hooks.append(h)
        return matching_hooks


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
