from ddqn import Ddqn

import numpy as np
from collections import deque
import random

class DdqnExperienceReplay(Ddqn):
    def __init__(self, env):
        Ddqn.__init__(self, env)
        self.save_path = 'app/saves/experience_replay'
        self.replay_buffer = deque(maxlen=20000)
        self.batch_size = 32

    def state_reshape(self, state):
        return state

    def get_action(self, state, greedy=False):
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if (np.random.random() < self.epsilon) and not greedy:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.network_predict(self.train_network, state.reshape(1, -1))[0])

        return action

    def update(self, state, action, next_state, reward, done):
        self.replay_buffer.append([state, action, next_state, reward, done])

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        samples = random.sample(self.replay_buffer, self.batch_size)

        states = []
        next_states = []
        for sample in samples:
            state, action, next_state, reward, done = sample
            states.append(state)
            next_states.append(next_state)

        states = np.reshape(states, (self.batch_size, self.env.observation_space.shape[0]))
        next_states = np.reshape(next_states, (self.batch_size, self.env.observation_space.shape[0]))
        
        targets = self.network_predict(self.train_network, states)
        new_targets = self.network_predict(self.target_network, next_states)

        for i, sample in enumerate(samples):
            state, action, next_state, reward, done = sample
            target = targets[i]

            if done:
                target[action] = reward
            else:
                Q_next = max(new_targets[i])
                target[action] = reward + self.gamma * Q_next

        self.network_fit(self.train_network, states, targets)

    def done_update(self, episode, score):
        self.replay()
        Ddqn.done_update(self, episode, score)