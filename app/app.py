import csv
import gym
import matplotlib.pyplot as plt
import pandas as pd

from dqn import Dqn
from ddqn import Ddqn
from ddqn_experience_replay import DdqnExperienceReplay
from ddqn_experience_replay_prioritized import DdqnExperienceReplayPrioritized

def run_episode(env, agent, load_path=None):
    state = env.reset()
    agent.load_network(load_path)

    done = False
    while not done:
        env.render()
        action = agent.get_action(state, greedy=True)
        state, _, done, _ = env.step(action)
    env.close()

def plot_stats(stats, smoothing_window=10):
    fig = plt.figure(figsize=(10,5))
    # stats_smoothed = pd.Series(stats).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # plt.plot(stats_smoothed)
    plt.plot(stats)
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")
    plt.show()

if __name__ == '__main__':
    # env = gym.make('CartPole-v1')
    # agent = DdqnExperienceReplayPrioritized(env)
    # agent = Dqn(env)
    # agent.train(episodes_num=1000)
    # load_path = 'app/saves/experience_replay_prioritized/episode700.h5'
    # run_episode(env, agent, load_path=load_path)

    df = pd.read_csv('app/saves/dqn/scores.csv', header=None, names=['sc'])
    plot_stats(df['sc'])