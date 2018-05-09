import tensorflow as tf
from model import ActorCritic
from environment import ParallelEnvironment
import numpy as np
from time import time

tf.enable_eager_execution()

NUM_PROCESSES = 8
EP_LENGTH = 100
NUM_EPOCHS = 500
ENV_NAME = 'Skiing-v0'


def collect_rollout(env, agent, max_t):
    ep_obs, ep_rew, ep_act, ep_vals = [], [], [], []
    obs = env.reset()

    for n in range(max_t):
        actions, values = agent(obs)
        ep_obs.append(obs)
        ep_act.append(actions)
        ep_vals.append(values)
        obs, rewards, dones, _ = env.step(actions)
        ep_rew.append(rewards)

    ep_obs = np.asarray(ep_obs, dtype=np.float32).swapaxes(1, 0).reshape((env.num_env * max_t,) + env.obs_shape)
    ep_rew = np.asarray(ep_rew, dtype=np.float32).swapaxes(1, 0)
    ep_act = np.asarray(ep_act, dtype=np.int32).swapaxes(1, 0)
    ep_vals = np.asarray(ep_vals, dtype=np.float32).swapaxes(1, 0)


    # _, last_values = agent(obs)
    #
    # for n, (rewards, ep_dones, value) in enumerate(zip(ep_rew, ep_dones, last_values)):
    #     rewards = rewards.tolist()
    #     ep_dones = ep_dones.tolist()
    #     if ep_dones[-1] == 0:
    #
    #         rewards = discount_with_dones(rewards + [value], ep_dones + [0], gamma)[:-1]
    #     else:
    #         rewards = discount_with_dones(rewards, ep_dones, gamma)
    #     ep_rew[n] = rewards
    #
    #
    # ep_rew = ep_rew.flatten()
    # ep_act = ep_act.flatten()
    # ep_vals = ep_vals.flatten()
    # mb_masks = mb_masks.flatten()

    return ep_obs, ep_rew, ep_act, ep_vals


def train(env, agent, epochs):
    for i in range(epochs):
        t1 = time()
        print("Epoch number: {}".format(i + 1))
        obs, rews, acts, vals = collect_rollout(env, agent, EP_LENGTH)
        agent.learn(obs, rews, acts, vals)
        print("Epoch completed in: {}\n\n".format(time() - t1))


def main():
    remote_envs = ParallelEnvironment(ENV_NAME, NUM_PROCESSES)
    agent = ActorCritic(remote_envs.actions)
    train(remote_envs, agent, NUM_EPOCHS)
    remote_envs.close()


if __name__ == '__main__':
    main()
