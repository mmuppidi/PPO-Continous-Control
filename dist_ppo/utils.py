
import math
import numpy as np
import torch
import torch.nn.functional as F


def log_policy_prob(mean, std, actions):
    # policy log probability
    act_log_softmax = -((mean-actions)**2)/(2*torch.exp(std).clamp(min=1e-4)) - torch.log(torch.sqrt(2*math.pi*torch.exp(std)))
    return act_log_softmax


def compute_log_policy_prob(memories, nn_policy, device):
    '''
    Run the policy on the observation in the memory and compute the policy log probability
    '''
    n_mean = nn_policy(torch.tensor(np.array([m.obs for m in memories], dtype=np.float32)).to(device))
    n_mean = n_mean.type(torch.DoubleTensor)
    logstd = nn_policy.logstd.type(torch.DoubleTensor)

    actions = torch.DoubleTensor(np.array([m.action for m in memories])).to(device)

    return log_policy_prob(n_mean, logstd, actions)


def clipped_PPO_loss(memories, nn_policy, nn_value, old_log_policy, adv, epsilon, device):
    '''
    Clipped PPO loss as in the paperself.
    It return the clipped policy loss and the value loss
    '''

    # state value
    rewards = torch.tensor(np.array([m.reward for m in memories], dtype=np.float32)).to(device)
    value = nn_value(torch.tensor(np.array([m.obs for m in memories], dtype=np.float32)).to(device))
    # Value loss
    vl_loss = F.mse_loss(value.squeeze(-1), rewards)

    new_log_policy = compute_log_policy_prob(memories, nn_policy, device)
    rt_theta = torch.exp(new_log_policy - old_log_policy.detach())

    adv = adv.unsqueeze(-1) # add a dimension because rt_theta has shape: [batch_size, n_actions]
    pg_loss = -torch.mean(torch.min(rt_theta*adv, torch.clamp(rt_theta, 1-epsilon, 1+epsilon)*adv))

    return pg_loss, vl_loss



