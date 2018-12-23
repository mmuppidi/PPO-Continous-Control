

import numpy as np
import os
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import A2C_policy, A2C_value
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    @classmethod
    def from_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        agent = cls(checkpoint_path['state_size'], checkpoint_path['action_size'])
        agent.agent_policy.load_state_dict(checkpoint['agent_policy'])
        agent.agent_value.load_state_dict(checkpoint['agent_value'])
        agent.optimizer_policy.load_state_dict(checkpoint['optimizer_policy'])
        agent.optimizer_value.load_state_dict(checkpoint['optimizer_value'])

        return agent


    def __init__(self, state_size, action_size, hyper_params, seed=0):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hyper_params = hyper_params
        # self.seed = random.seed(seed)

        # initialize the actor-critic NN
        self.agent_policy = A2C_policy(state_size, action_size).to(device)
        self.agent_value = A2C_value(state_size).to(device)

        # initialize policy and value optimizer
        self.optimizer_policy = optim.Adam(self.agent_policy.parameters(), lr=self.hyper_params['policy_lr'])
        self.optimizer_value = optim.Adam(self.agent_value.parameters(), lr=self.hyper_params['value_lr'])

        self.step_count = 0

        self.logs = []
        self.rollingmean_reward = None

    def step(self, state, action, action_theta, reward, next_state, done):
        pass

    def act(self, state):
        # get the agent policy
        ag_mean = self.agent_policy(torch.tensor(state))

        # get an action following the policy distribution
        logstd = self.agent_policy.logstd.data.cpu().numpy()
        action = ag_mean.data.cpu().numpy() + np.exp(logstd) * np.random.normal(size=logstd.shape)
        action = np.clip(action, -1, 1)

        state_value = float(self.agent_value(torch.tensor(state)))

        return action, state_value

    def _clipped_ppo_loss(self, batch, old_log_policy, adv):
        return utils.clipped_PPO_loss(
            batch, self.agent_policy, self.agent_value,
            old_log_policy, adv, self.hyper_params['clip_eps'], device)

    def learn(self, batch, gamma):
        # Compute the policy probability with the old policy network
        old_log_policy = utils.compute_log_policy_prob(batch, self.agent_policy, device)

        # Gather the advantage from the memory..
        batch_adv = np.array([m.adv for m in batch])
        # .. and normalize it to stabilize network
        batch_adv = (batch_adv - np.mean(batch_adv)) / (np.std(batch_adv) + 1e-7)
        batch_adv = torch.tensor(batch_adv).to(device)

        # variables to accumulate losses
        pol_loss_acc = []
        val_loss_acc = []

        # execute PPO_EPOCHS epochs
        for _ in range(self.hyper_params['ppo_epochs']):
            # compute the loss and optimize over mini batches of size BATCH_SIZE
            for mb in range(0, len(batch), self.hyper_params['batch_size']):
                mini_batch = batch[mb:mb+self.hyper_params['batch_size']]
                minib_old_log_policy = old_log_policy[mb:mb+self.hyper_params['batch_size']]
                minib_adv = batch_adv[mb:mb+self.hyper_params['batch_size']]

                # Compute the PPO clipped loss and the value loss
                pol_loss, val_loss = self._clipped_ppo_loss(mini_batch, minib_old_log_policy, minib_adv)

                # optimize the policy network
                self.optimizer_policy.zero_grad()
                pol_loss.backward()
                self.optimizer_policy.step()

                # optimize the value network
                self.optimizer_value.zero_grad()
                val_loss.backward()
                self.optimizer_value.step()

                pol_loss_acc.append(float(pol_loss))
                val_loss_acc.append(float(val_loss))

        if self.step_count % 100 == 0:
            print('step: {}; pg_loss: {}; vl_loss: {}'.format(self.step_count, np.mean(pol_loss_acc), np.mean(val_loss_acc)))
        self.logs.append({
            'pg_loss': np.mean(pol_loss_acc),
            'vl_loss': np.mean(val_loss_acc)
        })

        self.step_count += 1

    def save(self, rollingmean):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

        if self.rollingmean_reward:
            if (self.rollingmean_reward + (abs(self.rollingmean_reward)*0.1)) > rollingmean:
                return

        self.rollingmean_reward = rollingmean

        torch.save({
                    'agent_policy': self.agent_policy.state_dict(),
                    'agent_value': self.agent_value.state_dict(),
                    'optimizer_policy': self.optimizer_policy.state_dict(),
                    'optimizer_value': self.optimizer_value.state_dict(),
                    'action_size': self.action_size,
                    'state_size': self.state_size
                }, 'checkpoints/checkpoint.pth.tar')


class TrainedAgent():

    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.agent_policy = A2C_policy(checkpoint['state_size'], checkpoint['action_size']).to(device)
        self.agent_policy.load_state_dict(checkpoint['agent_policy'])

    def get_action(self, obs):
        ag_mean = self.agent_policy(torch.tensor(obs))
        action = np.clip(ag_mean.data.cpu().numpy().squeeze(), -1, 1)

        return action
