import gym
import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def make_discrete_env(gym_id, seed, idx, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_continuous_env(gym_id, seed, idx, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(0)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_buffers(envs, config, device):
    obs = torch.zeros((config['num_steps'], config['num_envs']) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config['num_steps'], config['num_envs']) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    rewards = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    dones = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    values = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(config['num_envs']).to(device)
    return obs, actions, logprobs, rewards, dones, values, next_obs, next_done