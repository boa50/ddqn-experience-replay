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
    rewards = 0

    done = False
    while not done:
        env.render()
        action = agent.get_action(state, greedy=True)
        state, reward, done, _ = env.step(action)
        rewards += reward
    env.close()

    print('Recompensa: {:3.0f}'.format(rewards))

def plot_stats(stats, smoothing_window=1, xlabel="", ylabel="", title=""):
    fig = plt.figure(figsize=(10,5))
    if smoothing_window > 1:
        stats_smoothed = pd.Series(stats).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(stats_smoothed)
    else:
        plt.plot(stats)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = DdqnExperienceReplayPrioritized(env)
    agent.train(episodes_num=1000)
    # load_path = 'app/saves/experience_replay_prioritized/episode800.h5'
    # run_episode(env, agent, load_path=load_path)
    
    # df = pd.read_csv('app/saves/dqn/metrics.csv')
    # plot_stats(df['exec_time'], xlabel="x", ylabel="y", title="title")