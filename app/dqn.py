from pathlib import Path
import itertools
import os
import csv

import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

class Dqn:
    def __init__(self, env):
        self.save_path = 'app/saves/dqn'
        self.env = env
        self.learningRate = 0.001
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.target_network = self.create_network()

    def state_reshape(self, state):
        return state.reshape(1, -1)

    def create_network(self):
        model = models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(layers.Dense(24, activation='relu', input_shape=state_shape))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learningRate))
        return model

    def network_predict(self, network, state):
        return network.predict(self.state_reshape(state))

    def network_fit(self, network, state, targets):
        network.fit(self.state_reshape(state), targets, epochs=1, verbose=0)

    def load_network(self, load_path=None):
        if load_path:
            self.target_network = models.load_model(load_path)
    
    def get_action(self, state, greedy=False):
        if (np.random.random() < self.epsilon) and not greedy:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.network_predict(self.target_network, state)[0])

        return action

    def update(self, state, action, next_state, reward, done):
        targets = self.network_predict(self.target_network, state)
        new_targets = self.network_predict(self.target_network, next_state)

        if done:
            targets[0][action] = reward
        else:
            Q_next = max(new_targets[0])
            targets[0][action] = reward + self.gamma * Q_next

        self.network_fit(self.target_network, state, targets)

    def done_update(self, episode, score):
        print('Episódio {:4d} \t pontuação = {:4d} \t epsilon = {:1.6f}'.format(episode, score, self.epsilon))
        self.score_save(score)

    def score_save(self, score):
        path = self.save_path + '/scores.csv'
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])

    def train(self, episodes_num):
        path = Path(self.save_path)
        if not path.exists():
            path.mkdir(parents=True)

        for episode in range(1, episodes_num + 1):
            state = self.env.reset()
            
            for i in itertools.count(0, 1):
                # self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action, next_state, reward, done)
                state = next_state

                if done:
                    self.done_update(episode, i)
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode % 100 == 0:
                self.target_network.save(str(path / ('episode' + str(episode) + '.h5')))

        env.close()