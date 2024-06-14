import typing


class EventDispatcher:
    def __init__(self, events: typing.List[str]):
        self.event_handlers: typing.Dict[str, typing.List[typing.Callable]] = {event_name: [] for event_name in events}

    def register_event(self, event_name: str):
        self.event_handlers[event_name] = []

    def __dispatch__(self, event_name: str, *args):
        for event_handler in self.event_handlers[event_name]:
            event_handler(*args)

    def add_event_handler(self, event_name: str, handler: typing.Callable):
        if event_name not in self.event_handlers:
            raise f"Event {event_name} not registered"
        self.event_handlers[event_name].append(handler)

