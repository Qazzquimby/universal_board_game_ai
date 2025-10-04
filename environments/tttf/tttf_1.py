import inspect
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Protocol, Callable

from environments.old_base import (
    BaseEnvironment,
    GameEntity,
    Selector,
    StateType,
    ActionResult,
    StateWithKey,
    ActionType,
)


def halve(number: int):
    return int(math.ceil(number / 2))


def action(default=False, targeter: Callable[["Hero", "TTTF_1"], List["Hero"]] = None):
    """
    Decorator to mark a method as an action and provide a targeter function.
    The targeter function is responsible for returning a list of valid targets.
    When applied to a method on a Hero subclass, it will be discoverable by
    `get_legal_actions`.
    """

    def decorator(func):
        func.is_default = default
        func.is_action = True
        func.targeter = targeter
        return func

    return decorator


def target_any_living_hero(actor: "Hero", env: "TTTF_1") -> List["Hero"]:
    """Targeter that returns all living heroes."""
    return [h for h in env.heroes if h.is_alive()]


@dataclass
class DamageEvent:
    """Event representing damage dealt to a hero."""

    actor: "Hero"
    target: "Hero"
    amount: int

    def resolve(self):
        self.target.health -= self.amount


@dataclass
class EndOfTurnEvent:
    """Event that fires at the end of a hero's turn."""

    actor: "Hero"  # The hero whose turn is ending

    def resolve(self):
        # This event is just a trigger, doesn't need to do anything on its own.
        pass


class DamageProto(Protocol):
    def __call__(
        self,
        actor: "Hero",
        target: "Hero",
        amount: int,
    ) -> None:
        ...


class EndOfTurnProto(Protocol):
    def __call__(self, actor: "Hero") -> None:
        ...


class TTTF_1(BaseEnvironment):
    """
    A simple team-based tactical environment using the game engine.
    - Two players, each with N heroes.
    - Turn-based, alternating between players' heroes.
    - Last team with a hero standing wins.
    """

    metadata = {"name": "tttf_1"}

    def __init__(
        self,
        num_players: int = 2,
        num_heroes_per_player: int = 2,
    ):
        super().__init__()
        assert num_players == 2, "TTTF_1 currently only supports 2 players."
        self._num_players = num_players
        self._num_heroes_per_player = num_heroes_per_player

        self.heroes: List[Hero] = []
        self.turn_order_ids: List[int] = []
        self.current_turn_index: int = 0
        self.winner: Optional[int] = None

        self.damage: DamageProto = self.define_action("damage", DamageEvent)
        self.end_of_turn: EndOfTurnProto = self.define_action(
            "end_of_turn", EndOfTurnEvent
        )

        self.reset()

    @property
    def num_players(self) -> int:
        return self._num_players

    def _reset(self) -> StateWithKey:
        # Clear engine-related state, but preserve action definitions
        self.triggered_abilities.clear()
        self.entities.clear()
        self.game_log.clear()
        self.turn = 0
        self.event_stack.clear()
        self.is_processing = False

        self.heroes = []

        # Create heroes in alternating turn order
        for i in range(self._num_heroes_per_player):
            for j in range(self._num_players):
                player_id = j
                hero_name = f"p{player_id}_h{i}"
                hero = Axe(self, hero_name, player_id)
                self.heroes.append(hero)

        self.turn_order_ids = list(range(len(self.heroes)))
        self.current_turn_index = 0
        self.winner = None
        self.done = False

        return self.get_state_with_key()

    def _step(self, action: ActionType) -> ActionResult:
        acting_hero = self.heroes[self.turn_order_ids[self.current_turn_index]]
        acting_player_id = acting_hero.player_owner_id

        # todo make this more generalizable targeting logic.
        action_name, target_id = action
        target_hero = self.heroes[target_id]

        action_method = getattr(acting_hero, action_name, None)
        with self.action_context(actor=acting_hero, action=action_method):
            action_method(target_hero)
        self.end_of_turn(actor=acting_hero)
        self._check_for_winner()

        reward = 0.0
        if self.done:
            if self.winner == acting_player_id:
                reward = 1.0
            elif self.winner is not None:
                reward = -1.0
        else:
            self._advance_turn()

        return ActionResult(
            next_state_with_key=self.get_state_with_key(),
            reward=reward,
            done=self.done,
        )

    def _advance_turn(self):
        num_heroes = len(self.heroes)
        for i in range(1, num_heroes + 1):
            next_index = (self.current_turn_index + i) % num_heroes
            acting_hero_id = self.turn_order_ids[next_index]
            if self.heroes[acting_hero_id].is_alive():
                self.current_turn_index = next_index
                return

        # All heroes are dead
        self.done = True

    def _check_for_winner(self):
        player_heroes_alive = [0] * self._num_players
        for hero in self.heroes:
            if hero.is_alive():
                player_heroes_alive[hero.player_owner_id] += 1

        alive_players = [p for p, count in enumerate(player_heroes_alive) if count > 0]

        if len(alive_players) <= 1:
            self.done = True
            self.winner = alive_players[0] if alive_players else None

    def get_legal_actions(self) -> List[ActionType]:
        assert not self.done

        # todo pass turn should always be a legal action

        acting_hero = self.heroes[self.turn_order_ids[self.current_turn_index]]
        return acting_hero.get_legal_actions()

    def get_current_player(self) -> int:
        # Before a move, current_turn_index points to the currently acting hero.
        acting_hero_id = self.turn_order_ids[self.current_turn_index]
        return self.heroes[acting_hero_id].player_owner_id

    def _get_state(self) -> StateType:
        # this should be nested, showing each entities state.
        # Needs to have same engine state as well, which was part of why I expected engine might be merged with BaseEnvironment
        return {
            "current_turn_index": self.current_turn_index,
            "done": self.done,
            "winner": self.winner,
        }

    def get_winning_player(self) -> Optional[int]:
        return self.winner

    def copy(self) -> "TTTF_1":
        new_env = TTTF_1(
            num_players=self._num_players,
            num_heroes_per_player=self._num_heroes_per_player,
        )
        new_env.set_state(self._get_state())
        return new_env

    def set_state(self, state: StateType) -> None:
        self._reset()

        # this should be nested, showing each entities state.
        self.current_turn_index = state["current_turn_index"]
        self.done = state["done"]
        self.winner = state["winner"]

        self._dirty = True

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            print("-" * 20)
            if self.done:
                print(f"Game Over! Winner: Player {self.winner}")
            else:
                acting_hero = self.heroes[self.turn_order_ids[self.current_turn_index]]
                print(
                    f"Turn: {self.current_turn_index}, "
                    f"Player {self.get_current_player()}'s turn ({acting_hero.name} acts)"
                )

            print("Heroes:")
            for i, hero in enumerate(self.heroes):
                status = "Alive" if hero.is_alive() else "Dead"
                print(
                    f"  ID {i}: {hero.name} (P{hero.player_owner_id}) - "
                    f"HP: {hero.health}/{hero.max_health} [{status}]"
                )
            print("-" * 20)


class Hero(GameEntity):
    """Represents a hero character in the game."""

    def __init__(self, env: TTTF_1, name: str, player_owner_id: int, max_health: int):
        super().__init__(env=env, name=name)
        self.env = env
        self.player_owner_id = player_owner_id
        self.max_health = max_health

        self.health = self.max_health

        # Discover methods decorated with @action
        self.actions = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_is_action"):
                self.actions[name] = method

    def get_legal_actions(self) -> List[ActionType]:
        """
        Returns a list of legal actions for this hero by checking all decorated
        @action methods and their targeters.
        """
        legal_actions: List[ActionType] = []
        hero_to_id = {hero: i for i, hero in enumerate(self.env.heroes)}

        for action_name, method in self.actions.items():
            targeter = getattr(method, "_targeter", None)
            if not targeter:
                # Action has no targets, e.g. a self-buff
                # legal_actions.append((action_name, None)) # Not supported yet
                continue

            valid_targets = targeter(self, self.env)
            for target in valid_targets:
                target_id = hero_to_id.get(target)
                if target_id is not None:
                    legal_actions.append((action_name, target_id))
        return legal_actions

    def is_alive(self) -> bool:
        return self.health > 0

    def __repr__(self):
        return f"<Hero {self.name}|p{self.player_owner_id}|hp:{self.health}>"


# todo
# start_of_turn = Selector(
#     "when start of my turn",


im_targeted = Selector(
    "when owner is target",
    lambda owner, event_target: event_target is owner,
    event_attr="target",
)

im_targeted_by_default_ability = Selector(
    "when owner is target of default ability",
    lambda owner, event_target: event_target is owner,
    event_attr="target",
    event_props={
        "cause.action.is_default": True
    },  # todo get from context _cause action
)


class Axe(Hero):
    def __init__(self, env: TTTF_1, name: str, player_owner_id: int):
        super().__init__(
            env=env,
            name=name,
            player_owner_id=player_owner_id,
            max_health=12,
        )

        self.after(
            im_targeted,
            self.env.damage,
            self._if_2plus_dmg__1dmg_aoe,
        )

        self.before(
            im_targeted_by_default_ability,
            self.env.damage,
            self._on_damaged_by_default_ability,
        )

    def _if_2plus_dmg__1dmg_aoe(self, event: DamageEvent):
        if event.amount >= 2:
            print(f"-> {self.name}'s passive: deals 1 damage to all enemies.")
            for hero in self.env.heroes:
                is_enemy = hero.player_owner_id != self.player_owner_id
                if is_enemy and hero.is_alive():
                    self.env.damage(actor=self, target=hero, amount=1)

    def _on_damaged_by_default_ability(self, event: DamageEvent):
        """Reflects half of incoming damage back to the attacker."""
        reflected_damage = halve(event.amount)
        if reflected_damage > 0:
            print(
                f"-> {self.name}'s passive: reflects {reflected_damage} damage to {event.actor.name}!"
            )
            self.env.damage(actor=self, target=event.actor, amount=reflected_damage)

    @action(default=True, targeter=target_any_living_hero)
    def axe_attack(self, target: Hero):
        self.env.damage(actor=self, target=target, amount=2)

    @action(targeter=target_any_living_hero)
    def battle_hunger(self, target: Hero):
        print(f"-> {self.name} applies Battle Hunger to {target.name}")

        def take_1_damage(event: EndOfTurnEvent):
            print(f"-> {target.name} takes 1 damage from {self.name}'s Battle Hunger.")
            self.env.damage(actor=self, target=target, amount=1)

        self.after(target, self.env.end_of_turn, take_1_damage)


class Lina(Hero):
    def __init__(self, env: TTTF_1, name: str, player_owner_id: int):
        super().__init__(
            env=env,
            name=name,
            player_owner_id=player_owner_id,
            max_health=6,
        )
        self.fiery_soul_charges = 0

    @action(default=True, targeter=target_any_living_hero)
    def lina_attack(self, target: Hero):
        self.env.damage(
            actor=self,
            target=target,
            amount=1 + self.fiery_soul_charges,
        )

    @action(targeter=target_any_living_hero)
    def dragon_slave(self, target: Hero):
        self.env.damage(
            actor=self,
            target=target,
            amount=1 + self.fiery_soul_charges,
        )
        # 1/game
        # target gets dot token
        # self.fiery_soul_charges += 1 # todo probably a gain token event

    @action(targeter=target_any_living_hero)
    def light_strike_array(self, target: Hero):
        self.env.damage(
            actor=self,
            target=target,
            amount=2 + self.fiery_soul_charges,
        )
        # 1/game
        # "tap one of their abilities" - prevent using basic abilities till next turn?
        # self.fiery_soul_charges += 1 # todo probably a gain token event

    @action(targeter=target_any_living_hero)
    def laguna_blade(self, target: Hero):
        self.env.damage(
            actor=self,
            target=target,
            amount=3 * self.fiery_soul_charges,
        )
        # 1/game
        # self.fiery_soul_charges = 2 # todo...


class Necrophos(Hero):
    def __init__(self, env: TTTF_1, name: str, player_owner_id: int):
        super().__init__(
            env=env,
            name=name,
            player_owner_id=player_owner_id,
            max_health=8,
        )
        # start of turn, enemies take irreducible 1 + kill counters // 2, and you heal killcounters
        # on kill, if they had 4 or more health ,gain a kill counter

    @action(default=True, targeter=target_any_living_hero)
    def necrophos_attack(self, target: Hero):
        pass

        # enemies take 1 damage
        # you and allies hael 1

    # Ghost Shroud
    # 1/Game, Instant +3
    # Until the end of your next turn:
    #   You cannot be affected by default abilities.
    #   You deal +100% healing.
    #   You receive +1 damage.
    # Death Seeker
    # 1/Game
    # Teleport to a space adjacent to an enemy in range 3.
    # Use a default ability.
    # Reaper's Scythe
    # 1/Game
    # Range 3, immobilize.
    # At the start of your next turn, deal irreducible damage the target equal to their missing health.
    # On kill, gain 2 additional Kill counters.


if __name__ == "__main__":
    # Setup and run a 2v2 game of all Axes
    env = TTTF_1(num_heroes_per_player=2)
    env.render()

    while not env.done:
        legal_actions = env.get_legal_actions()
        chosen_action = random.choice(legal_actions)
        acting_hero = env.heroes[env.turn_order_ids[env.current_turn_index]]
        target_hero = env.heroes[chosen_action[1]]
        print(
            f"\n*** Player {env.get_current_player()} ({acting_hero.name}) "
            f"uses {chosen_action[0]} on {target_hero.name} ***"
        )

        env.step(chosen_action)
        env.render()

    print("\n--- FINAL ---")
    env.render()
