import numpy as np
from collections import namedtuple

import gym

Memory = namedtuple('Memory', ['obs', 'action', 'new_obs', 'reward', 'done', 'value', 'adv'], verbose=False, rename=False)


class UnityEnv(object):

    game_rew = 0
    last_game_rew = 0
    game_n = 0
    last_games_rews = [0]
    n_iter = 0

    def __init__(self, env, n_steps, gamma, gae_lambda):
        super(UnityEnv, self).__init__()

        # create the new environment
        self.env = env

        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # reset the environment
        env_info = self.env.reset(train_mode=True)[self.brain_name]

        # number of agents
        self.num_agents = len(env_info.agents)

        # size of each action
        self.action_n = self.brain.vector_action_space_size

        # examine the state space
        self.obs = env_info.vector_observations

        self.n_steps = n_steps
        self.action_n = self.brain.vector_action_space_size
        self.observation_n = env_info.vector_observations.shape[1]

        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def get_actions(self, agent):
        return zip(*[agent.act(obs) for obs in self.obs])

    # CHANGED
    def steps(self, agent):
        '''
        Execute the agent n_steps in the environment
        '''
        memories = [[] for i in range(self.num_agents)]
        #self.game_rew = [0 for i in range(self.num_agents)]

        for s in range(self.n_steps):
            self.n_iter += 1

            # get action and state_value from the agent for the current obs
            actions, state_values = self.get_actions(agent)

            # Perform a step in the environment
            # new_obs, reward, done, _ = self.env.step(action)

            env_info = self.env.step(np.array(actions))[self.brain_name]

            new_obs = env_info.vector_observations

            rewards = env_info.rewards
            dones = env_info.local_done

            # Update the memories with the last interaction
            if any(dones):
                for i in range(self.num_agents):
                    # change the reward to 0 in case the episode is end
                    memories[i].append(
                        Memory(
                            obs=self.obs[i], action=actions[i], new_obs=new_obs[i],
                            reward=0, done=dones[i], value=state_values[i], adv=0
                        )
                    )

            else:
                for i in range(self.num_agents):
                    memories[i].append(
                        Memory(
                            obs=self.obs[i], action=actions[i], new_obs=new_obs[i],
                            reward=rewards[i], done=dones[i], value=state_values[i], adv=0
                        )
                    )

            self.game_rew += np.mean(rewards)
            self.obs = new_obs

            if any(dones):
                # print('#####',self.game_n, 'rew:', int(self.game_rew), int(np.mean(self.last_games_rews[-100:])), np.round(reward, 2), self.n_iter)

                # reset the environment
                env_info = self.env.reset(train_mode=True)[self.brain_name]
                self.obs = env_info.vector_observations
                self.last_game_rew = self.game_rew
                self.game_rew = 0
                self.game_n += 1
                self.n_iter = 0
                self.last_games_rews.append(self.last_game_rew)

        # compute the discount reward of the memories and return it
        return self.generalized_advantage_estimation(memories)

    def generalized_advantage_estimation(self, memories):
        mem = []
        for memory in memories:
            mem += self._generalized_advantage_estimation(memory)

        return mem

    def _generalized_advantage_estimation(self, memories):
        '''
        Calculate the advantage diuscounted reward as in the paper
        '''
        upd_memories = []
        run_add = 0

        for t in reversed(range(len(memories)-1)):
            if memories[t].done:
                run_add = memories[t].reward
            else:
                sigma = memories[t].reward + self.gamma * memories[t+1].value - memories[t].value
                run_add = sigma + run_add * self.gamma * self.gae_lambda

            ## NB: the last memoy is missing
            # Update the memories with the discounted reward
            upd_memories.append(Memory(obs=memories[t].obs, action=memories[t].action, new_obs=memories[t].new_obs, reward=run_add + memories[t].value, done=memories[t].done, value=memories[t].value, adv=run_add))

        return upd_memories[::-1]
