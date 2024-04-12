import typing
from .mouse import MouseObservation
class Reward(object):
    def __init__(self,
                 reward_structure: dict):
        self.reward_structure = reward_structure

    def __call__(self, observation: typing.List[float]) -> float:
        reward = 0.0
        for field, multiplier in self.reward_structure.items():
            reward += multiplier * observation[MouseObservation.Field[field].value]
        return reward
