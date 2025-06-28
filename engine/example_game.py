from dataclasses import dataclass

from game_engine import GameEngine, GameEntity

game = GameEngine()


class Player(GameEntity):
    def __init__(self, game: GameEngine, name):
        super().__init__(game=game, id_=name)  # todo how would you typically name id?
        self.amount = 10

    def __repr__(self):
        return f"<Player {self.id}|{self.amount}>"


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
    event.player.amount += event.amount


def _resolve_lose(event: LoseEvent):
    event.player.amount -= event.amount


def _resolve_turn_start(event: TurnStartEvent):
    game.turn += 1
    print(f"*** TURN {game.turn} (Player: {event.player.id}) ***")


gain = game.define_action("gain", GainEvent, _resolve_gain)
lose = game.define_action("lose", LoseEvent, _resolve_lose)
start_turn = game.define_action("start_turn", TurnStartEvent, _resolve_turn_start)


this_player = lambda owner, event: event.player == owner
another_player = lambda owner, event: hasattr(event, "player") and event.player != owner


if __name__ == "__main__":
    p1 = Player(game, "Alice")
    p2 = Player(game, "Bob")

    print("--- Setting up abilities ---")

    def _gain_3_on_turn_start(event: TurnStartEvent):
        gain(player=event.player, amount=3)

    p1.after(this_player, start_turn, _gain_3_on_turn_start)

    def _gain_twice(event: GainEvent):
        event.amount *= 2

    p1.modify(this_player, gain, _gain_twice)

    def _gain_instead_of_lose(event: LoseEvent):
        gain(player=event.player, amount=event.amount)

    p1.replace(this_player, lose, _gain_instead_of_lose)

    def _gain_1_less(event: GainEvent):
        event.amount = max(0, event.amount - 1)

    p1.modify(another_player, gain, _gain_1_less)

    # Bob just gets the basic turn start ability.
    p2.after(this_player, start_turn, _gain_3_on_turn_start)

    # --- Game Start ---
    print("\n\n--- STARTING GAME ---")
    print(f"Initial State: {p1}, {p2}")

    for i in range(5):
        start_turn(player=p1)
        start_turn(player=p2)

    print("p1 arbitrary penalty")
    lose(player=p1, amount=5)

    print(f"\nFinal State: {p1}, {p2}")
