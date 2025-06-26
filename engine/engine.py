# self.after(enter_battlefield)


class Entity:
    pass


class Player(Entity):
    pass


class Action:
    pass


class Event:
    pass


# Reaction = function(evt)


class Engine:
    def __init__(self):
        self.event_queue = []
        self.listeners = {}

    def after(self, actor, event, do):
        pass


gain_player = Player
