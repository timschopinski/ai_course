import numpy as np
from rl_base import Agent, Action, State
import os


class QAgent(Agent):

    def __init__(self, n_states, n_actions,
                 name='QAgent', initial_q_value=0.0, q_table=None):
        super().__init__(name)

        # hyperparams
        # TODO ustaw te parametry na sensowne wartości
        self.lr = 0.15  # learning rate
        self.gamma = 0.92  # discount factor
        self.epsilon = 0.1  # exploration rate
        self.eps_decrement = 0.001  # exploration rate decrement
        self.eps_min = 0.01  # minimum exploration rate

        self.action_space = [i for i in range(n_actions)]
        self.n_states = n_states
        self.q_table = q_table if q_table is not None else self.init_q_table(initial_q_value)

    def init_q_table(self, initial_q_value=0.):
        # TODO - utwórz tablicę wartości Q o rozmiarze [n_states, n_actions] wypełnioną początkowo wartościami initial_q_value
        q_table = np.full((self.n_states, len(self.action_space)), initial_q_value)
        return q_table

    def update_action_policy(self) -> None:
        # TODO - zaktualizuj wartość epsilon
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_decrement

    def choose_action(self, state: State) -> Action:

        assert 0 <= state < self.n_states, \
            f"Bad state_idx. Has to be int between 0 and {self.n_states}"

        # TODO - zaimplementuj strategię eps-zachłanną
        if np.random.uniform() > self.epsilon:
            action = np.argmax(self.q_table[state])
        else:
            action = np.random.choice(self.action_space)

        return Action(action)

    def learn(self, state: State, action: Action, reward: float, new_state: State, done: bool) -> None:
        # TODO - zaktualizuj q_table
        q_value = self.q_table[state][action]
        max_q_value = np.max(self.q_table[new_state])

        if done:
            target = reward
        else:
            target = reward + self.gamma * max_q_value

        self.q_table[state][action] = q_value + self.lr * (target - q_value)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

    def get_instruction_string(self):
        return [f"Linearly decreasing eps-greedy: eps={self.epsilon:0.4f}"]
