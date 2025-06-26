from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Callable, Any

# ==============================================================================
# 1. Core Data Structures & Engine
# ==============================================================================

# Internal representation of a registered ability
Hook = namedtuple(
    "Hook", ["hook_type", "target_filter", "action_name", "handler", "owner"]
)

# A LogEntry stores the final state of a resolved event
LogEntry = namedtuple("LogEntry", ["action_name", "event_data", "turn"])


class GameEngine:
    def __init__(self):
        self.players = {}

        self.action_resolvers = {}
        self.hooks = []  # Probably want to make this into one list per action type
        self.event_stack = deque()
        self.game_log = []
        self.turn = 0
        self.is_processing = False

    def add_player(self, player):
        self.players[player.name] = player
        player.engine = self

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

            # 1. REPLACEMENT PHASE
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

            # 2. MODIFICATION PHASE
            for hook in self._get_hooks_for_event("modify", action_name, event):
                print(f"  - Applying 'modify' hook from {hook.owner.name}'s ability")
                # The handler now mutates the event object directly.
                hook.handler(event)
                print(f"    -> Event modified to: {event}")

            # 3. RESOLUTION PHASE
            print(f"  - Resolving: {action_name}({event})")
            resolver = self.action_resolvers.get(action_name)
            if resolver:
                resolver(event)
            else:
                print(f"    -> WARNING: No resolver found for action '{action_name}'")
            self.game_log.append(LogEntry(action_name, event, self.turn))

            # 4. AFTER PHASE
            for hook in self._get_hooks_for_event("after", action_name, event):
                print(f"  - Applying 'after' hook from {hook.owner.name}'s ability")
                hook.handler(event)

            chain_depth -= 1

        self.is_processing = False
        print("\n--- Event queue empty ---")

    def _get_hooks_for_event(self, hook_type, action_name, event):
        """Finds hooks using the new flexible target filters."""
        matching_hooks = []
        for h in self.hooks:
            if h.hook_type == hook_type and h.action_name == action_name:
                # The target is now a callable filter function.
                if h.target_filter(h.owner, event):
                    matching_hooks.append(h)
        return matching_hooks


# ==============================================================================
# 2. Scripter-Facing API
# ==============================================================================

# Global engine instance
engine = GameEngine()


class Player:
    def __init__(self, name):
        self.name = name
        self.amount = 10
        self.engine = None

    def __repr__(self):
        return f"<Player {self.name}|{self.amount}>"


# --- Define Actions ---
# This part would be in the core game setup, not necessarily the ability scripts.


@dataclass
class GainEvent:
    player: Player
    amount: int


@dataclass
class LoseEvent:
    player: Player
    amount: int


@dataclass
class TurnStartEvent:
    player: Player


def _resolve_gain(event: GainEvent):
    if event.amount > 0:
        event.player.amount += event.amount


def _resolve_lose(event: LoseEvent):
    if event.amount > 0:
        event.player.amount -= event.amount


def _resolve_turn_start(event: TurnStartEvent):
    engine.turn += 1
    print(f"*** TURN {engine.turn} (Player: {event.player.name}) ***")


gain = engine.define_action("gain", GainEvent, _resolve_gain)
lose = engine.define_action("lose", LoseEvent, _resolve_lose)
start_turn = engine.define_action("start_turn", TurnStartEvent, _resolve_turn_start)


# --- Define Abilities ---
def add_ability(owner, hook):
    engine.hooks.append(Hook(**dict(hook._asdict(), owner=owner)))


def modify(target_filter, action, handler):
    return Hook("modify", target_filter, action.__name__, handler, owner=None)


def replace(target_filter, action, handler):
    return Hook("replace", target_filter, action.__name__, handler, owner=None)


def after(target_filter, action, handler):
    return Hook("after", target_filter, action.__name__, handler, owner=None)


# --- Define Target Filters ---
this_player = lambda owner, event: event.player == owner
another_player = lambda owner, event: hasattr(event, "player") and event.player != owner

# --- Game Log Query ---
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


def get(action):
    results = [log for log in engine.game_log if log.action_name == action.__name__]
    return QuerySet(results)


# ==============================================================================
# 3. Example Game Script
# ==============================================================================

if __name__ == "__main__":
    p1 = Player("Alice")
    p2 = Player("Bob")
    engine.add_player(p1)
    engine.add_player(p2)

    print("--- Setting up abilities ---")

    # Ability 1: At the start of your turn, gain 3. (Using a local handler)
    def _gain_3_on_turn_start(event: TurnStartEvent):
        gain(player=event.player, amount=3)

    add_ability(p1, after(this_player, start_turn, _gain_3_on_turn_start))

    # Ability 2: When you would gain, gain twice that amount. (Mutating handler)
    def _gain_twice(event: GainEvent):
        event.amount *= 2

    add_ability(p1, modify(this_player, gain, _gain_twice))

    # Ability 3: When you would lose, instead gain that much. (Replacing handler)
    def _gain_instead_of_lose(event: LoseEvent):
        gain(player=event.player, amount=event.amount)

    add_ability(p1, replace(this_player, lose, _gain_instead_of_lose))

    # Ability 4: When someone else would gain, they gain one less. (Local lambda-like handler)
    def _weaken_gain(event: GainEvent):
        event.amount = max(0, event.amount - 1)

    add_ability(p1, modify(another_player, gain, _weaken_gain))

    # Bob just gets the basic turn start ability.
    add_ability(p2, after(this_player, start_turn, _gain_3_on_turn_start))

    # --- Game Start ---
    print("\n\n--- STARTING GAME ---")
    print(f"Initial State: {p1}, {p2}")

    for i in range(5):
        start_turn(player=p1)
        start_turn(player=p2)

    print("p1 arbitrary penalty")
    lose(player=p1, amount=5)

    print(f"\nFinal State: {p1}, {p2}")
