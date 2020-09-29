import gym

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

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    # agent = DdqnExperienceReplayPrioritized(env)
    agent = Dqn(env)
    agent.train(episodes_num=1000)
    # load_path = 'app/saves/experience_replay_prioritized/episode700.h5'
    # run_episode(env, agent, load_path=load_path)