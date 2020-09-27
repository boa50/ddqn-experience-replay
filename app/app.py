from pathlib import Path
import itertools
import os
import csv
from collections import deque
import random

import gym
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

class EnvTrain:
    def __init__(self, env):
        self.env = env
        self.learningRate = 0.001
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.train_network = self.create_network()
        self.target_network = self.create_network()
        self.target_network.set_weights(self.train_network.get_weights())
        ### Experience Replay
        self.replay_buffer = deque(maxlen=20000)
        self.batch_size = 32
        ####

    def state_reshape(self, state):
        ### Experience Replay
        return state
        ####
        # return state.reshape(1, -1)

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
        network.fit(self.state_reshape(state), targets, epochs=0, verbose=1)

    def load_network(self, load_path=None):
        if load_path:
            self.train_network = models.load_model(load_path)
            self.target_network = models.load_model(load_path)
    
    def get_action(self, state, greedy=False):
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if (np.random.random() < self.epsilon) and not greedy:
            action = self.env.action_space.sample()
        else:
            ### Experience Replay
            action = np.argmax(self.network_predict(self.train_network, state.reshape(1, -1))[0])
            ####
            ### DDQN
            # action = np.argmax(self.network_predict(self.train_network, state)[0])
            ####
            # action = np.argmax(self.network_predict(self.target_network, state)[0])

        return action

    def update(self, state, action, next_state, reward, done):
        ### Experience Replay
        self.replay_buffer.append([state, action, next_state, reward, done])

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
        ####

        # ### DDQN
        # targets = self.network_predict(self.train_network, state)
        # ####
        # # targets = self.network_predict(self.target_network, state)
        # new_targets = self.network_predict(self.target_network, next_state)

        # if done:
        #     targets[0][action] = reward
        # else:
        #     Q_next = max(new_targets[0])
        #     targets[0][action] = reward + self.gamma * Q_next

        # ### DDQN
        # self.network_fit(self.train_network, state, targets)
        # ####
        # # self.network_fit(self.target_network, state, targets)

    def train(self, episodes_num):
        for episode in range(1, episodes_num + 1):
            state = self.env.reset()
            
            for i in itertools.count(0, 1):
                # self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action, next_state, reward, done)
                state = next_state

                if done:
                    print('Episódio {:4d} \t pontuação = {:4d} \t epsilon = {:1.6f}'.format(episode, i, self.epsilon))
                    score_save(i)
                    ### DDQN
                    self.target_network.set_weights(self.train_network.get_weights())
                    ###
                    break

            self.epsilon *= self.epsilon_decay

            if episode % 100 == 0:
                path = Path('app/saves')
                if not path.exists():
                    path.mkdir(parents=True)
                self.target_network.save(str(path / ('episode' + str(episode) + '.h5')))

        env.close()

def score_save(score):
    path = 'app/saves/scores.csv'
    if not os.path.exists(path):
        with open(path, "w"):
            pass
    scores_file = open(path, "a")
    with scores_file:
        writer = csv.writer(scores_file)
        writer.writerow([score])

def run_episode(env, agent, load_path=None):
    state = env.reset()
    agent.load_network(load_path)

    done = False
    while not done:
        env.render()
        action = agent.get_action(state, greedy=True)
        state, reward, done, _ = env.step(action)
    env.close()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = EnvTrain(env)
    # agent.train(episodes_num=1000)
    load_path = 'app/saves/episode500.h5'
    run_episode(env, agent)