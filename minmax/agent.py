from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, token):
        self.my_token = token
        self.opponent_token = 'o' if token == 'x' else 'x'

    @abstractmethod
    def decide(self, connect4):
        ...
