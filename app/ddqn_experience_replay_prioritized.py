from ddqn_experience_replay import DdqnExperienceReplay

from collections import deque
import random

import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


class DdqnExperienceReplayPrioritized(DdqnExperienceReplay):
    def __init__(self, env):
        DdqnExperienceReplay.__init__(self, env)
        self.save_path = 'app/saves/experience_replay_prioritized'
        self.priority = deque(maxlen=20000)
        self.alpha = 0.6
        self.importance = 1

    def importance_loss(self):
        def loss(y_true, y_pred):
            return tf.reduce_mean(tf.multiply(tf.square(y_pred - y_true), self.importance))

        return loss

    def create_network(self):
        model = models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(layers.Dense(24, activation='relu', input_shape=state_shape))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss=self.importance_loss(), optimizer=Adam(lr=self.learningRate))
        return model
    
    def buffer_insert(self, state, action, next_state, reward, done):
        target_prediction = np.max(self.network_predict(self.target_network, next_state.reshape(1, -1))[0])
        q_next = reward + self.gamma * target_prediction
        q = self.network_predict(self.train_network, state.reshape(1, -1))[0][action]
        p = (np.abs(q_next - q) + (np.e ** -10)) ** self.alpha
        self.priority.append(p)
        self.replay_buffer.append((state, action, next_state, reward, done))

    def update(self, state, action, next_state, reward, done):
        self.buffer_insert(state, action, next_state, reward, done)

    def get_priority_experience_batch(self):
        p_sum = np.sum(self.priority)
        prob = self.priority / p_sum
        sample_indices = random.choices(range(len(prob)), k=self.batch_size, weights=prob)
        importances = (1/prob) * (1/len(self.priority))
        importances = np.array(importances)[sample_indices]
        samples = np.array(self.replay_buffer)[sample_indices]
        return samples, importances

    def replay(self):
        samples, importances = self.get_priority_experience_batch()
        for sample, importance in zip(samples, importances):
            state, action, next_state, reward, done = sample
            target = reward
            if not done:
                target_prediction = np.max(self.network_predict(self.target_network, next_state.reshape(1, -1))[0])
                target = reward + self.gamma * target_prediction
            final_target = self.network_predict(self.train_network, state.reshape(1, -1))
            final_target[0][action] = target
            imp = importance ** (1 - self.epsilon)
            # imp = np.reshape(imp, 1)
            self.importance = imp
            self.network_fit(self.train_network, state.reshape(1, -1), final_target)