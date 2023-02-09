import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from common.utils import layer_init


class PPODiscreteAgent(nn.Module):

    def __init__(self, envs):
        super().__init__()

        self.action_space = envs.single_action_space.n
        self.observation_space = np.array(envs.single_observation_space.shape).prod()

        self.critic = nn.Sequential(layer_init(nn.Linear(self.observation_space, 64)),
                                    nn.Tanh(),
                                    layer_init(nn.Linear(64, 64)),
                                    nn.Tanh(),
                                    layer_init(nn.Linear(64, 1), std=1.))

        self.actor = nn.Sequential(layer_init(nn.Linear(self.observation_space, 64)),
                                   nn.Tanh(),
                                   layer_init(nn.Linear(64, 64)),
                                   nn.Tanh(),
                                   layer_init(nn.Linear(64, self.action_space), std=0.01))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
