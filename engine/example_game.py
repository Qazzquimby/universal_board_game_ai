from dataclasses import dataclass

from engine.engine import GameEngine, Hook

engine = GameEngine()


class Player:
    def __init__(self, name):
        self.name = name
        self.amount = 10
        self.engine = None

    def __repr__(self):
        return f"<Player {self.name}|{self.amount}>"


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

# TODO replace all this with engine.modify, etc. Should include the add_ability call
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
