from dqn import Dqn

import numpy as np

class Ddqn(Dqn):
    def __init__(self, env):
        Dqn.__init__(self, env)
        self.save_path = 'app/saves/ddqn'
        self.train_network = self.create_network()
        self.target_network = self.create_network()
        self.target_network.set_weights(self.train_network.get_weights())

    def get_action(self, state, greedy=False):
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if (np.random.random() < self.epsilon) and not greedy:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.network_predict(self.train_network, state)[0])

        return action

    def update(self, state, action, next_state, reward, done):
        targets = self.network_predict(self.train_network, state)
        new_targets = self.network_predict(self.target_network, next_state)

        if done:
            targets[0][action] = reward
        else:
            Q_next = max(new_targets[0])
            targets[0][action] = reward + self.gamma * Q_next

        self.network_fit(self.train_network, state, targets)

    def done_update(self, episode, score):
        Dqn.done_update(self, episode, score)
        self.target_network.set_weights(self.train_network.get_weights())