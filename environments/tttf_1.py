from dataclasses import dataclass
from typing import List, Tuple, Optional, Protocol

from environments.base import (
    BaseEnvironment,
    StateType,
    SanityCheckState,
    ActionResult,
    StateWithKey,
    ActionType,
)
from engine.game_engine import GameEngine, GameEntity


@dataclass
class DamageEvent:
    """Event representing damage dealt to a hero."""

    actor: "Hero"
    target: "Hero"
    amount: int

    def resolve(self):
        self.target.health -= self.amount


class DamageProto(Protocol):
    def __call__(self, actor: "Hero", target: "Hero", amount: int) -> None:
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
        hero_health: int = 10,
    ):
        super().__init__()
        assert num_players == 2, "TTTF_1 currently only supports 2 players."
        self._num_players = num_players
        self._num_heroes_per_player = num_heroes_per_player
        self._hero_health = hero_health

        self.engine: Optional[GameEngine] = None
        self.heroes: List[Hero] = []
        self.turn_order_ids: List[int] = []
        self.current_turn_index: int = 0
        self.winner: Optional[int] = None
        self.engine = GameEngine()

        self.reset()

        self.damage: DamageProto = self.engine.define_action("damage", DamageEvent)

    @property
    def num_players(self) -> int:
        return self._num_players

    # TODO later, replace this with one tensor size per entity type, including player, and one for Game
    # Will be input to transformer architecture
    @property
    def observation_tensor_shape(self) -> Tuple[int, ...]:
        # Vector: [h1_hp, h2_hp, ..., hn_hp]
        num_features = len(self.heroes)
        return (num_features,)

    # TODO later, replace this with one head per verb and choice type
    @property
    def policy_vector_size(self) -> int:
        # Max possible actions: attack any of the heroes.
        return len(self.heroes)

    def map_action_to_policy_index(self, action: ActionType) -> Optional[int]:
        return action

    def map_policy_index_to_action(self, index: int) -> Optional[ActionType]:
        return index

    def _reset(self) -> StateWithKey:
        self.engine = GameEngine()
        self.heroes = []

        # Create heroes in alternating turn order
        for i in range(self._num_heroes_per_player):
            for j in range(self._num_players):
                player_id = j
                hero_name = f"p{player_id}_h{i}"
                hero = Axe(self, hero_name, player_id, self._hero_health)
                self.heroes.append(hero)

        self.turn_order_ids = list(range(len(self.heroes)))
        self.current_turn_index = 0
        self.winner = None
        self.done = False

        return self.get_state_with_key()

    def _step(self, action: ActionType) -> ActionResult:
        if self.done:
            return ActionResult(
                next_state_with_key=self.get_state_with_key(),
                reward=0.0,
                done=True,
            )

        acting_hero = self.heroes[self.turn_order_ids[self.current_turn_index]]
        acting_player_id = acting_hero.player_owner_id

        target_hero = self.heroes[action]
        acting_hero.attack(target_hero)

        self.engine.run()
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

        acting_hero = self.heroes[self.turn_order_ids[self.current_turn_index]]
        return acting_hero.get_legal_actions()

    def get_current_player(self) -> int:
        # Before a move, current_turn_index points to the currently acting hero.
        acting_hero_id = self.turn_order_ids[self.current_turn_index]
        return self.heroes[acting_hero_id].player_owner_id

    def _get_state(self) -> StateType:
        return {
            "hero_healths": [h.health for h in self.heroes],
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
            hero_health=self._hero_health,
        )
        new_env.set_state(self._get_state())
        return new_env

    def set_state(self, state: StateType) -> None:
        # To restore state, we need to rebuild the hero list and engine state.
        # A full reset is easiest to ensure engine is clean.
        self._reset()

        # Apply the saved state
        healths = state["hero_healths"]
        for i, hero in enumerate(self.heroes):
            hero.health = healths[i]

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
        super().__init__(engine=env.engine, name=name)
        self.env = env
        self.player_owner_id = player_owner_id
        self.max_health = max_health

        self.health = self.max_health

    def get_legal_actions(self) -> List[ActionType]:
        """
        Returns a list of legal actions for this hero.
        Base hero has no actions. Subclasses should override this.
        """
        return []

    def attack(self, target: "Hero"):
        """
        Performs an attack on a target.
        Base hero cannot attack. Subclasses should override this.
        """
        raise NotImplementedError(
            f"Hero {self.name} of class {self.__class__.__name__} does not have an 'attack' action."
        )

    def is_alive(self) -> bool:
        return self.health > 0

    def __repr__(self):
        return f"<Hero {self.name}|p{self.player_owner_id}|hp:{self.health}>"


class Axe(Hero):
    def __init__(self, env: TTTF_1, name: str, player_owner_id: int, max_health: int):
        super().__init__(
            env=env,
            name=name,
            player_owner_id=player_owner_id,
            max_health=max_health,
        )

    def get_legal_actions(self) -> List[ActionType]:
        """Axe can attack any living enemy hero."""
        legal_targets = []
        for i, hero in enumerate(self.env.heroes):
            is_enemy = hero.player_owner_id != self.player_owner_id
            if is_enemy and hero.is_alive():
                legal_targets.append(i)  # Action is target hero ID
        return legal_targets

    def attack(self, target: Hero):
        """Axe's attack deals 1 damage to the target."""
        self.env.damage(actor=self, target=target, amount=1)
