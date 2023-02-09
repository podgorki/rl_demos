import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from common.utils import layer_init


class PPOContinuousAgent(nn.Module):

    def __init__(self, envs):
        super().__init__()

        self.action_space = np.array(envs.single_action_space.shape).prod()
        self.observation_space = np.array(envs.single_observation_space.shape).prod()

        self.critic = nn.Sequential(layer_init(nn.Linear(self.observation_space, 64)),
                                    nn.Tanh(),
                                    layer_init(nn.Linear(64, 64)),
                                    nn.Tanh(),
                                    layer_init(nn.Linear(64, 1), std=1.))

        self.actor_mean = nn.Sequential(layer_init(nn.Linear(self.observation_space, 64)),
                                        nn.Tanh(),
                                        layer_init(nn.Linear(64, 64)),
                                        nn.Tanh(),
                                        layer_init(nn.Linear(64, self.action_space), std=0.01))

        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_space))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
