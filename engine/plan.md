





Integrate

from typing import Type, Callable, Any

class Selector:
    """An object that represents a query for a target, not a specific target."""
    def __init__(self, base_type: Type, description: str, filter_func: Callable[[Any], bool]):
        self.base_type = base_type
        self.description = description  # For debugging and logging
        self.filter_func = filter_func

    def matches(self, instance: Any) -> bool:
        """Checks if a given instance matches the selector's criteria."""
        # Fast path: if it's not even an instance of the base type, fail.
        if not isinstance(instance, self.base_type):
            return False
        # If it is the right type, apply the detailed filter logic.
        return self.filter_func(instance)
    
    def __repr__(self) -> str:
        return f"<Selector: {self.description}>"



# In your game's 'event_selectors.py' or similar

# --- For a Player-based game ---
def any_player_except(me: 'Player') -> Selector:
    """Selects any Player instance that is not 'me'."""
    return Selector(
        base_type=Player,
        description=f"any Player except {me.name}",
        filter_func=lambda target: target is not me
    )

def an_opponent_of(me: 'Player') -> Selector:
    """Selects any Player instance on a different team from 'me'."""
    return Selector(
        base_type=Player,
        description=f"any opponent of {me.name}",
        filter_func=lambda target: hasattr(target, 'team') and target.team != me.team
    )

# --- For a Card game ---
def another_card_than(this_card: 'Card') -> Selector:
    """Selects any Card instance that is not 'this_card'."""
    return Selector(
        base_type=Card,
        description=f"any Card other than {this_card.name}",
        filter_func=lambda target: target is not this_card
    )

def any_creature() -> Selector:
    """Selects any Card that has the 'Creature' subtype."""
    return Selector(
        base_type=Card,
        description="any Creature",
        filter_func=lambda target: "Creature" in target.subtypes
    )



Use overloads to make selector optional

@overload
def when(event_name: str) -> EventQuery: ...

@overload
def when(selector: Type | any | SelectorFunction, event_name: str) -> EventQuery: ...


