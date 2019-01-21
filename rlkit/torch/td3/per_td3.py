from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm, np_to_pytorch_batch

from rlkit.torch.td3.td3 import TD3
from rlkit.data_management.baselines_per_buffer import BaselinesPERBuffer
from baselines.common.schedules import LinearSchedule

class PERTD3(TD3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = BaselinesPERBuffer(self.replay_buffer_size, alpha=0.6)
        self.t = 0
        total_train_steps = 100 * 10000
        self.beta_schedule = LinearSchedule(total_train_steps, initial_p=0.4, final_p=1.0)

    def get_batch(self):
        batch = self.replay_buffer.random_batch(
                self.batch_size, beta=self.beta_schedule.value(self.t)
                )
        obs, act, rew, next_obs, done, weights, idxes = batch
        batch = {
            'rewards': rew,
            'terminals': done,
            'observations': obs,
            'actions': act,
            'next_observations': next_obs
        }
        return np_to_pytorch_batch(batch), weights, idxes

    def _do_training(self):
        batch, weights, idxes = self.get_batch()
        self.t += 1
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        weights = torch.autograd.Variable(ptu.from_numpy(
            weights.reshape(weights.shape + (1,))
            ), requires_grad=False)

        """
        Critic operations.
        """

        next_actions = self.target_policy(next_obs)
        noise = torch.normal(
            torch.zeros_like(next_actions),
            self.target_policy_noise,
        )
        noise = torch.clamp(
            noise,
            -self.target_policy_noise_clip,
            self.target_policy_noise_clip
        )
        noisy_next_actions = next_actions + noise

        target_q1_values = self.target_qf1(next_obs, noisy_next_actions)
        target_q2_values = self.target_qf2(next_obs, noisy_next_actions)
        target_q_values = torch.min(target_q1_values, target_q2_values)
        q_target = rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        q1_pred = self.qf1(obs, actions)
        bellman_errors_1 = (q1_pred - q_target) ** 2

        # IS 
        bellman_errors_1 = weights * bellman_errors_1

        qf1_loss = bellman_errors_1.mean()

        q2_pred = self.qf2(obs, actions)
        bellman_errors_2 = (q2_pred - q_target) ** 2

        # IS
        bellman_errors_2 = weights * bellman_errors_2

        qf2_loss = bellman_errors_2.mean()

        """
        Update Networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        # Update priorities
        EPS = 10 ** -6
        td_error_1 = np.abs(ptu.get_numpy(q1_pred) - q_target)
        td_error_2 = np.abs(ptu.get_numpy(q2_pred) - q_target)
        new_priorities = (td_error_1 + td_error_2) / 2 + EPS
        self.replay_buffer.update_priorities(idxes, new_priorities)

        policy_actions = policy_loss = None
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            policy_actions = self.policy(obs)
            q_output = self.qf1(obs, policy_actions)
            policy_loss = - q_output.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.tau)

        """
        Save some statistics for eval using just one batch.
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            if policy_loss is None:
                policy_actions = self.policy(obs)
                q_output = self.qf1(obs, policy_actions)
                policy_loss = - q_output.mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 1',
                ptu.get_numpy(bellman_errors_1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 2',
                ptu.get_numpy(bellman_errors_2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))
