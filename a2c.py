import tensorflow as tf
from model import ActorCritic
from environment import ParallelEnvironment
import numpy as np
from utils import discount_with_dones

tf.enable_eager_execution()

NUM_PROCESSES = 8
EP_LENGTH = 100
NUM_EPOCHS = 500
ENV_NAME = 'Skiing-v0'


def collect_rollout(env, agent, initial_obs, max_t, gamma=0.99):
    roll_obs, roll_rew, roll_act, roll_vals, roll_dones = [], [], [], [], []
    obs = initial_obs
    dones = [False for _ in range(env.num_env)]

    for n in range(max_t):
        actions, values = agent(obs)
        roll_obs.append(obs)
        roll_act.append(actions)
        roll_vals.append(values)
        roll_dones.append(dones)
        obs, rewards, dones, _ = env.step(actions)
        roll_rew.append(rewards)

    roll_obs = np.asarray(roll_obs, dtype=np.float32).swapaxes(1, 0).reshape((env.num_env * max_t,) + env.obs_shape)
    roll_rew = np.asarray(roll_rew, dtype=np.float32).swapaxes(1, 0)
    roll_act = np.asarray(roll_act, dtype=np.int32).swapaxes(1, 0)
    roll_vals = np.asarray(roll_vals, dtype=np.float32).swapaxes(1, 0)
    _, last_values = agent(obs)

    for n, (rewards, ep_dones, value) in enumerate(zip(roll_rew, roll_dones, last_values)):
        rewards = rewards.tolist()
        ep_dones = ep_dones.tolist()
        if ep_dones[-1] == 0:

            rewards = discount_with_dones(rewards + [value], ep_dones + [0], gamma)[:-1]
        else:
            rewards = discount_with_dones(rewards, ep_dones, gamma)
        roll_rew[n] = rewards


    roll_rew = roll_rew.flatten()
    roll_act = roll_act.flatten()
    roll_vals = roll_vals.flatten()


    return obs, roll_obs, roll_rew, roll_act, roll_vals


def train(env, agent, epochs):
    initial_obs = env.reset()
    for i in range(epochs):
        initial_obs, obs, rews, acts, vals = collect_rollout(env, agent, initial_obs, EP_LENGTH)
        agent.learn(obs, rews, acts, vals)


def main():
    remote_envs = ParallelEnvironment(ENV_NAME, NUM_PROCESSES)
    agent = ActorCritic(remote_envs.actions)
    train(remote_envs, agent, NUM_EPOCHS)
    remote_envs.close()


if __name__ == '__main__':
    main()
