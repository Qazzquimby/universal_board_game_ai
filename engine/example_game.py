from asyncio import Protocol
from dataclasses import dataclass

from game_engine import GameEngine, GameEntity, Selector

game = GameEngine()


class Player(GameEntity):
    def __init__(self, engine: GameEngine, name: str):
        # A descriptive string like `name` is often better for an entity's primary
        # identifier than a generic `id`, unless it's a UUID or integer.
        super().__init__(engine=engine, name=name)
        self.amount = 10

    def __repr__(self):
        return f"<Player {self.name}|{self.amount}>"


@dataclass
class GainEvent:
    actor: Player
    amount: int

    def resolve(self):
        self.actor.amount += self.amount


class GainProto(Protocol):
    def __call__(self, actor: Player, amount: int) -> None:
        ...


gain: GainProto = game.define_action("gain", GainEvent)


@dataclass
class LoseEvent:
    actor: Player
    amount: int

    def resolve(self):
        self.actor.amount -= self.amount


class LoseProto(Protocol):
    def __call__(self, actor: Player, amount: int) -> None:
        ...


lose: LoseProto = game.define_action("lose", LoseEvent)


@dataclass
class StartTurnEvent:
    actor: Player

    def resolve(self):
        game.turn += 1  # todo is this global?
        print(f"*** TURN {game.turn} (Player: {self.actor.name}) ***")


class StartTurnProto(Protocol):
    def __call__(self, actor: Player) -> None:
        ...


start_turn = game.define_action("start_turn", StartTurnEvent)


this_player = Selector("target is owner", lambda owner, target: target is owner)
another_player = Selector(
    "target is not owner",
    lambda owner, target: target is not None and target is not owner,
)


if __name__ == "__main__":
    p1 = Player(game, "Alice")
    p2 = Player(game, "Bob")

    print("--- Setting up abilities ---")

    def _gain_3_on_turn_start(event: StartTurnEvent):
        gain(actor=event.actor, amount=3)

    p1.after(this_player, start_turn, _gain_3_on_turn_start)

    def _gain_twice(event: GainEvent):
        event.amount *= 2

    p1.modify(this_player, gain, _gain_twice)

    def _gain_instead_of_lose(event: LoseEvent):
        gain(actor=event.actor, amount=event.amount)

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
        start_turn(actor=p1)
        start_turn(actor=p2)

    print("p1 arbitrary penalty")
    lose(actor=p1, amount=5)

    print(f"\nFinal State: {p1}, {p2}")


###

print(GainEvent.__text_signature__)
