import numpy as np
import gym
from multiprocessing import Pipe, Process
import cv2


def worker(conn, env_id, seed):
    env = EnvWrapper(env_id)
    env.seed(seed)
    while True:
        cmd, data = conn.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            conn.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            conn.send(ob)
        elif cmd == 'close':
            conn.close()
            break
        elif cmd == 'get_spaces':
            conn.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class ParallelEnvironment:

    def __init__(self, env_id, num_env):
        parents, children = zip(*[Pipe() for _ in range(num_env)])
        processes = [Process(target=worker, args=(child_conn, env_id, n))
                     for n, child_conn in enumerate(children)]
        for p in processes:
            p.daemon = True
            p.start()

        parents[0].send(('get_spaces', None))

        def step(actions):
            for remote, action in zip(parents, actions):
                remote.send(('step', action))
            results = [parent.recv() for parent in parents]
            obs, rews, dones, infos = zip(*results)
            return np.stack(obs), np.stack(rews), np.stack(dones), infos

        def reset():
            for parent in parents:
                parent.send(('reset', None))
            return np.stack([remote.recv() for remote in parents])

        def close():
            for parent in parents:
                parent.send(('close', None))
                parent.close()
            for pr in processes:
                pr.join()

        self.step = step
        self.reset = reset
        self.close = close
        self.num_env = num_env
        self.obs_shape, self.actions = parents[0].recv()


class EnvWrapper:

    def __init__(self, env_id):
        self.env = gym.make(env_id)
        self.action_space = self.env.action_space.n
        self.observation_space = (42, 42, 1)

    def process_obs(self, obs):
        frame = cv2.resize(obs, (80, 80))
        frame = cv2.resize(frame, (42, 42))
        frame = frame.mean(2)
        frame = np.expand_dims(frame, 2)
        frame *= (1.0 / 255.0)
        return frame

    def reset(self):
        obs = self.env.reset()
        return self.process_obs(obs)

    def render(self):
        self.env.render()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            obs = self.env.reset()
        obs = self.process_obs(obs)
        return obs, rew, done, info

    def seed(self, seed):
        self.env.seed(seed)
