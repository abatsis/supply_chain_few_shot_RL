import argparse

import tqdm
import or_gym
from or_gym.utils import create_env
import json
from stable_baselines3 import PPO
from glob import glob
import os
import secrets
import numpy as np
import torch as th
from stable_baselines3.common.monitor import Monitor

th.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--skewed', action='store_true')
args, unknown = parser.parse_known_args()
args = vars(args)

def poison_rand():
    return {'mu': 70}


def binomial_rand():
    return {'n': np.random.randint(10, 100), 'p': np.random.uniform(0.3, 0.5)}


def uniform_rand():
    return {'low': 10.0, 'high': 100.0}


def geometric_rand():
    return {'p': np.random.uniform(0.3, 0.5)}


switch = {
    1: poison_rand(),
    2: binomial_rand(),
    3: uniform_rand(),
    4: geometric_rand(),
}


def create_random_env_config(distribution_choice, m):
    return dict(
        periods=np.random.randint(10, 100),
        I0=[np.random.randint(10, 100) for i in range(m - 1)],
        p=np.random.randint(2, 100),
        r=[np.random.random() * 2 for i in range(m)],
        k=[np.random.random() for i in range(m)],
        h=[np.random.random() * 0.5 for i in range(m - 1)],
        c=[np.random.randint(10, 100) for i in range(m - 1)],
        L=[np.random.randint(1, 20) for i in range(m - 1)],
        backlog=False,
        dist=distribution_choice,
        dist_param=switch[distribution_choice],
        alpha=0.97,
        seed_int=0,
    )


def create_random_env_config_skewed(distribution_choice, m):
    distribution_choice = 1
    return dict(
        periods=np.random.randint(10, 100),
        I0=[np.random.randint(10, 100) for i in range(m - 1)],
        p=np.random.randint(2, 100),
        r=[np.random.random() * 2 for i in range(m)],
        k=[np.random.random() for i in range(m)],
        h=[np.random.random() * 0.5 for i in range(m - 1)],
        c=[np.random.randint(10, 100) for i in range(m - 1)],
        L=[np.random.randint(1, 20) for i in range(m - 1)],
        backlog=False,
        dist=distribution_choice,
        dist_param=switch[distribution_choice],
        alpha=0.97,
        seed_int=0,
    )


def evaluate_using_checkpoint(model, num_episodes, env):
    obss = []
    actions = []
    rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        reward = 0.0
        done = False
        while not done:
            obss.append(obs)
            action, _ = model.predict(
                observation=obs,
                deterministic=True,
                # policy_id="default_policy",  # <- default value
            )
            # print(obs.shape,action.shape)
            obs, r, done, _ = env.step(action)
            # print([action])
            actions.append(action)
            reward += r
        rewards.append(reward)
    # print(np.mean(rewards))
    return np.array(obss), np.array(actions), rewards


env_name = 'InvManagement-v1'
m = np.random.randint(3, 10)
# d = np.random.choice([1,2,3,4])
distribution_choice = 3
env_config_generator = create_random_env_config_skewed if args['skewed'] else create_random_env_config
env_config = env_config_generator(distribution_choice, m)

print(env_config)

env = or_gym.make(env_name, env_config=env_config)
#env = Monitor(env, "./monitor")
policy_kwargs = dict(activation_fn=th.nn.ELU, net_arch=dict(pi=[256, 256], vf=[256, 256]))
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0)

episodes = 200
pbar = tqdm.tqdm(range(episodes))
for i in pbar:
    result = model.learn(total_timesteps=356)
    #pbar.set_description(f"Reward sum:{np.sum(rewards):10.3f}")
    #print(np.sum(rewards))

obs, actions, rewards = evaluate_using_checkpoint(model, 20, env)

data_dir = args['data_dir']
os.makedirs(data_dir, exist_ok=True)
file = data_dir + "/" + secrets.token_hex(nbytes=32) + ".npz"
np.savez(file, obs=obs, actions=actions, rewards=rewards, config=env_config)
