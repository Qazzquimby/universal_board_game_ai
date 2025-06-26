import inspect
from collections import deque, namedtuple

# ==============================================================================
# 1. Core Data Structures
# ==============================================================================

# A Hook is the internal representation of an ability.
Hook = namedtuple("Hook", ["hook_type", "target", "action_name", "handler", "owner"])

# A LogEntry stores what actually happened.
LogEntry = namedtuple("LogEntry", ["action_name", "params", "turn"])

# An Event is a request to perform an action, which travels through the system.
class Event:
    def __init__(self, action_name, params):
        self.action_name = action_name
        self.params = params

    def __repr__(self):
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.action_name}({params_str})"


# ==============================================================================
# 2. The Engine
# ==============================================================================


class GameEngine:
    def __init__(self):
        self.players = {}
        self.hooks = []
        self.event_stack = deque()
        self.game_log = []
        self.turn = 0
        self.is_processing = False

    def add_player(self, player):
        self.players[player.name] = player
        player.engine = self

    def run(self):
        self.is_processing = True
        chain_depth = 0
        MAX_DEPTH = 50

        while self.event_stack:
            if chain_depth > MAX_DEPTH:
                raise RuntimeError(
                    f"Event chain limit ({MAX_DEPTH}) exceeded. Potential infinite loop."
                )

            event = self.event_stack.popleft()
            print(f"\n--- Processing: {event} ---")
            chain_depth += 1

            # 1. REPLACEMENT PHASE
            # A replacement cancels the current event and puts a new one on the stack.
            replaced = False
            for hook in self._get_hooks_for_event("replace", event):
                print(f"  - Applying 'replace' hook from {hook.owner.name}'s ability")
                # We temporarily stop processing to see if the handler queues a new event.
                start_len = len(self.event_stack)
                hook.handler(event.params)
                if len(self.event_stack) > start_len:
                    print(f"    -> Event was REPLACED. New event is on the stack.")
                    replaced = True
                    break  # Only one replacement allowed per event

            if replaced:
                continue  # Restart the loop with the new event

            # 2. MODIFICATION PHASE
            # Modifiers serially update the event's parameters.
            for hook in self._get_hooks_for_event("modify", event):
                print(f"  - Applying 'modify' hook from {hook.owner.name}'s ability")
                new_params = hook.handler(dict(event.params))  # Pass a copy
                event.params.update(new_params)
                print(f"    -> Params modified to: {event.params}")

            # 3. RESOLUTION PHASE
            # The action's effect on the game state happens here.
            print(f"  - Resolving: {event}")
            self._resolve_action(event)
            self.game_log.append(LogEntry(event.action_name, event.params, self.turn))

            # 4. AFTER PHASE
            # These hooks trigger new, separate events.
            for hook in self._get_hooks_for_event("after", event):
                print(f"  - Applying 'after' hook from {hook.owner.name}'s ability")
                hook.handler(event.params)

            chain_depth -= 1

        self.is_processing = False
        print("\n--- Event queue empty ---")

    def _get_hooks_for_event(self, hook_type, event):
        matching_hooks = []
        for h in self.hooks:
            is_correct_type = h.hook_type == hook_type
            is_correct_action = h.action_name == event.action_name

            is_targeted_correctly = False
            if h.target == "self" and event.params.get("player") == h.owner:
                is_targeted_correctly = True
            elif h.target == "other" and event.params.get("player") != h.owner:
                is_targeted_correctly = True
            elif h.target == event.params.get("player"):  # Direct player object target
                is_targeted_correctly = True

            if is_correct_type and is_correct_action and is_targeted_correctly:
                matching_hooks.append(h)
        return matching_hooks

    def _resolve_action(self, event):
        """The hardcoded effects of base game actions."""
        if event.action_name == "gain":
            if event.params["amount"] > 0:
                event.params["player"].amount += event.params["amount"]
        elif event.action_name == "lose":
            if event.params["amount"] > 0:
                event.params["player"].amount -= event.params["amount"]
        elif event.action_name == "start_turn":
            self.turn += 1
            print(f"*** TURN {self.turn} (Player: {event.params['player'].name}) ***")


# ==============================================================================
# 3. Scripter-Facing API
# ==============================================================================

# Global engine instance for simplicity in scripting
engine = GameEngine()


class Player:
    def __init__(self, name):
        self.name = name
        self.amount = 10
        self.engine = None

    def __repr__(self):
        return f"<Player {self.name}|{self.amount}>"


# --- Action Functions ---
def _create_action(name):
    def action(**kwargs):
        event = Event(name, kwargs)
        engine.event_stack.append(event)
        # If the engine isn't running, start it.
        if not engine.is_processing:
            engine.run()

    return action


gain = _create_action("gain")
lose = _create_action("lose")
start_turn = _create_action("start_turn")

# --- Ability Definition Functions ---
def add_ability(owner, hook):
    # The hook functions below will return a partial Hook object.
    # We complete it here with the owner.
    full_hook = Hook(owner=owner, **hook._asdict())
    engine.hooks.append(full_hook)


def modify(target, action, handler):
    return Hook("modify", target, action.__name__, handler, owner=None)


def replace(target, action, handler):
    return Hook("replace", target, action.__name__, handler, owner=None)


def after(target, action, handler):
    return Hook("after", target, action.__name__, handler, owner=None)


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
# 4. Example Game Script
# ==============================================================================

# --- Handler Definitions ---
def gain_3_on_turn_start(params):
    gain(player=params["player"], amount=3)


def gain_twice(params):
    return {"amount": params["amount"] * 2}


def gain_instead_of_lose(params):
    gain(player=params["player"], amount=params["amount"])


def gain_1_less(params):
    return {"amount": max(0, params["amount"] - 1)}


# --- Setup ---
p1 = Player("Alice")
p2 = Player("Bob")
engine.add_player(p1)
engine.add_player(p2)

print("--- Setting up abilities ---")
# At the start of your turn, gain 3
add_ability(p1, after("self", start_turn, gain_3_on_turn_start))
# When you would gain, gain twice that amount
add_ability(p1, modify("self", gain, gain_twice))
# When you would lose, instead gain that much
add_ability(p1, replace("self", lose, gain_instead_of_lose))
# When someone else would gain, they gain one less
add_ability(p1, modify("other", gain, gain_1_less))

add_ability(p2, after("self", start_turn, gain_3_on_turn_start))

# --- Game Start ---
print("\n\n--- STARTING GAME ---")
print(f"Initial State: {p1}, {p2}")

start_turn(player=p1)
# Expected outcome for Alice's turn:
# 1. start_turn(p1) resolves.
# 2. 'after' hook triggers gain(p1, 3).
# 3. 'gain' event is processed.
# 4. 'modify' hook (gain_twice) fires. params['amount'] becomes 6.
# 5. gain(p1, 6) resolves. Alice's amount becomes 10 + 6 = 16.

start_turn(player=p2)
# Expected outcome for Bob's turn:
# 1. start_turn(p2) resolves.
# 2. 'after' hook triggers gain(p2, 3).
# 3. 'gain' event is processed.
# 4. p1's 'modify(other)' hook fires. params['amount'] becomes 3 - 1 = 2.
# 5. gain(p2, 2) resolves. Bob's amount becomes 10 + 2 = 12.

lose(player=p1, amount=5)
# Expected outcome for Alice losing:
# 1. lose(p1, 5) is processed.
# 2. 'replace' hook fires, calling gain(p1, 5). The lose event is cancelled.
# 3. gain(p1, 5) is processed.
# 4. 'modify' hook (gain_twice) fires. params['amount'] becomes 10.
# 5. gain(p1, 10) resolves. Alice's amount becomes 16 + 10 = 26.

print(f"\nFinal State: {p1}, {p2}")
