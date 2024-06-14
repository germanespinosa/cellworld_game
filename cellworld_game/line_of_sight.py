import typing
from .agent import Agent
from .event import EventDispatcher


class LineOfSight(EventDispatcher):
    def __init__(self):
        EventDispatcher.__init__(self, ["update"])
        self.state: typing.Dict[str, typing.Dict[str, bool]] = {}
        self.agents: typing.Dict[str, Agent] = {}

    def register_agent(self, agent: Agent):
        self.state[agent.name] = {}
        for other_agent_name in self.agents:
            self.register_event(f"{agent.name}_{other_agent_name}")
            self.register_event(f"{other_agent_name}_{agent.name}")
            self.state[agent.name][other_agent_name] = False
            self.state[other_agent_name][agent.name] = False
        self.register_event(f"{agent.name}_update")
        self.agents[agent.name] = agent

    def __setitem__(self,
                    key: tuple,
                    is_visible: bool):

        src_agent, dst_agent = key
        if is_visible:
            self.state[src_agent][dst_agent] = True
            self.__dispatch__(f"{src_agent}_{dst_agent}", self.agents[dst_agent].state)
        else:
            self.state[src_agent][dst_agent] = False

    def update_agent(self,
                     agent_name: str):
        self.__dispatch__(f"{agent_name}_update", self.state[agent_name])

    def update_all(self):
        self.__dispatch__(f"update", self.state)

    def __getitem__(self,
                    key: typing.Union[tuple, str]):
        if isinstance(key, tuple):
            src_agent, dst_agent = key
            return self.state[src_agent][dst_agent]
        else:
            return self.state[key]

