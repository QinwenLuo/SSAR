import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from tensorboardX import SummaryWriter
from dataclasses import fields, replace

from net import ParaNet
from action_selection import IQL_qv
from dataset_utils import ReplayBuffer, Online_ReplayBuffer
from utils import soft_update, compute_mean_std, normalize_states, return_reward_range, wrap_env, is_goal_reached, \
    set_seed, eval_actor, select_trustable_trajectories, traj_selection_for_dense_rewards

TensorBatch = List[torch.Tensor]

ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "hopper-medium-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    offline_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path

    # CQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 3e-5  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    target_update_period: int = 1  # Frequency of target nets updates
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = False  # Use Lagrange version of CQL
    cql_target_action_gap: float = -1.0  # Action gap
    cql_temp: float = 1.0  # CQL temperature
    cql_alpha: float = 10.0  # Minimal Q weight
    cql_max_target_backup: bool = False  # Use max target backup
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    q_n_hidden_layers: int = 3  # Number of hidden layers in Q networks
    reward_scale: float = 1.0  # Reward scale for normalization
    reward_bias: float = 0.0  # Reward bias for normalization

    # AntMaze hacks
    bc_steps: int = int(0)  # Number of BC steps at start
    policy_log_std_multiplier: float = 1.0

    """adaptive constraint params"""
    iql_tau: float = 0.5  # the hyperparameter of IQL, set the same value of IQL
    select_actions: bool = True  # whether to select high-value actions
    select_method: str = 'traj'  # the method to obtain the sub-dataset, use 'traj' or 'iql'
    ref_return: int = 3000  # the return threshold for obtain the sub-dataset with high returns
    loosen_bound: bool = False  # whether to apply the regularization on the sub-dataset with high-value actions
    min_n_away: float = 1.0  # the initial threshold n_{start}
    max_n_away: float = 3.0  # the maximum threshold n_{end}
    update_period: int = 50000  # the interval of updating the threshold n
    beta_lr: float = 1e-8  # the learning rate of coefficient net

    """offline-to-online setting"""
    online_timesteps: int = 250000  # total online steps
    warmup_steps: int = 5000  # steps for warm up
    decay_steps: int = 400000  # the decay steps for the coefficients
    load_way: str = 'part'  # load all/part/none data, or half sample
    load_offline: bool = False  # use offline pre-trained model as initialization


def load_train_config_auto(config):
    env_name_lower = "_".join(config.env.split("-")[:1]).lower().replace("-", "_")
    env_lower = "_".join(config.env.split("-")[1:]).lower().replace("-", "_")

    file_path = os.path.join(f"config/cql/{env_name_lower}", f"{env_lower}.yaml")

    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
        config_fields = fields(config)

        filtered_config_data = {field.name: config_data[field.name] for field in config_fields if
                                field.name in config_data}
        config = replace(config, **filtered_config_data)
        return config


def modify_reward(
        dataset: Dict,
        env_name: str,
        max_episode_steps: int = 1000,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
) -> Dict:
    modification_data = {}
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        modification_data = {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias
    return modification_data


def modify_reward_online(
        reward: float,
        env_name: str,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        **kwargs,
) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    reward = reward * reward_scale + reward_bias
    return reward


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Lasy layers should be initialzied differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
            self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
            self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(torch.tanh(mean), std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return action_distribution.log_prob(sample)

    def forward(
            self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(torch.tanh(mean), std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_action: float,
            log_std_multiplier: float = 1.0,
            log_std_offset: float = -1.0,
            orthogonal_init: bool = False,
            no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        init_module_weights(self.base_network)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
            self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(
            self,
            observations: torch.Tensor,
            deterministic: bool = False,
            repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()

    @torch.no_grad()
    def log_prob_away_from_mean(self, observations: torch.Tensor, n=1):
        with torch.no_grad():
            base_network_output = self.base_network(observations)
            mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
            log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
            log_std = torch.clamp(log_std, -20, 2.0)
            std = torch.exp(log_std)
            mean_actions = torch.tanh(mean)
            action_1 = torch.tanh((mean + n * std)).clamp(-0.99999, 0.99999)
            action_2 = torch.tanh((mean - n * std)).clamp(-0.99999, 0.99999)
            action_1 = torch.clamp(action_1, min=mean_actions - n * 0.1, max=mean_actions + n * 0.1)
            action_2 = torch.clamp(action_2, min=mean_actions - n * 0.1, max=mean_actions + n * 0.1)
            log_prob = torch.min(self.tanh_gaussian.log_prob(mean, log_std, action_1),
                                 self.tanh_gaussian.log_prob(mean, log_std, action_2))
        return log_prob

    @torch.no_grad()
    def get_mean_std(self, observations: torch.Tensor):
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, -20, 2.0)
        std = torch.exp(log_std)
        return mean, std


class FullyConnectedQFunction(nn.Module):
    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            orthogonal_init: bool = False,
            n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        layers = [
            nn.Linear(observation_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.LayerNorm(256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)

        init_module_weights(self.network, orthogonal_init)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class ContinuousCQL:
    def __init__(
            self,
            env,
            critic_1,
            critic_1_optimizer,
            critic_2,
            critic_2_optimizer,
            actor,
            actor_optimizer,
            target_entropy: float,
            discount: float = 0.99,
            alpha_multiplier: float = 1.0,
            use_automatic_entropy_tuning: bool = True,
            backup_entropy: bool = False,
            policy_lr: bool = 3e-4,
            qf_lr: bool = 3e-4,
            soft_target_update_rate: float = 5e-3,
            bc_steps=100000,
            target_update_period: int = 1,
            cql_n_actions: int = 10,
            cql_importance_sample: bool = True,
            cql_lagrange: bool = False,
            cql_target_action_gap: float = -1.0,
            cql_temp: float = 1.0,
            cql_alpha: float = 5.0,
            cql_max_target_backup: bool = False,
            cql_clip_diff_min: float = -np.inf,
            cql_clip_diff_max: float = np.inf,
            device: str = "cpu",
            min_n_away: float = 2.0,
            max_n_away: float = 3.0,
            update_period: int = 100000,
            beta_lr: float = 5e-9,
            decay_steps: int = 250000,
            select_actions: bool = True,
            loosen_bound: bool = True
    ):
        super().__init__()
        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device

        self.total_it = 0

        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.actor = actor

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self.total_it = 0

        # initialize the coefficient net
        self.beta = ParaNet(self.critic_1.observation_dim, cql_alpha).to(self._device)
        self.beta_optimizer = torch.optim.Adam(self.beta.parameters(), lr=beta_lr)

        # initialize the threshold
        self.update_threshold = True
        self.n_away = min_n_away
        self.max_n_away = max_n_away
        self.increase_step = (self.max_n_away - self.n_away) / 10
        self.update_period = update_period

        self.select_actions = select_actions
        self.loosen_bound = loosen_bound

        self.decay_steps = decay_steps

        self.env_ant = True if 'ant' in env else False

    def decay_factor(self, online_steps):
        return np.cos(min(online_steps / self.decay_steps, 1.0) * np.pi / 2)

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                    self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            new_actions: torch.Tensor,
            alpha: torch.Tensor,
            log_pi: torch.Tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions).sum(-1, keepdim=True)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            if not self.env_ant:
                policy_loss = (alpha * log_pi - q_new_actions).mean()
            else:
                bc_loss = ((new_actions - actions) ** 2).sum(-1)
                policy_loss = (alpha * log_pi - q_new_actions + 0.5 * bc_loss).mean()

        return policy_loss

    def _q_loss(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            next_observations: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            alpha: torch.Tensor,
            log_dict: Dict,
            trusts: torch.Tensor,
            online_steps: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(
                next_observations, repeat=self.cql_n_actions
            )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(
                next_log_pi, -1, max_target_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(
            observations, repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self.actor(
            next_observations, repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q1_predicted, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q2_predicted, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5 ** action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        )
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        )

        beta = self.beta(observations).detach()

        if self.loosen_bound:
            beta *= trusts

        if online_steps > 0:
            beta *= self.decay_factor(online_steps)

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                    alpha_prime
                    * (beta * (cql_qf1_diff - self.cql_target_action_gap)).mean()
            )
            cql_min_qf2_loss = (
                    alpha_prime
                    * (beta * (cql_qf2_diff - self.cql_target_action_gap)).mean()
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = (cql_qf1_diff * beta).mean()
            cql_min_qf2_loss = (cql_qf2_diff * beta).mean()
            alpha_prime_loss = observations.new_tensor(0.0)
            alpha_prime = observations.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(
            dict(
                qf_loss=qf_loss.item(),
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
            )
        )

        log_dict.update(
            dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
            )
        )

        return qf_loss, alpha_prime, alpha_prime_loss

    def train(self, batch: TensorBatch, online_steps: int = 0) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            trusts,
        ) = batch
        trusts = trusts.squeeze(-1)

        self.total_it += 1

        log_dict = {}

        new_actions, log_pi = self.actor(observations)

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        """ Policy loss """
        policy_loss = self._policy_loss(
            observations, actions, new_actions, alpha, log_pi
        )

        _, std = self.actor.get_mean_std(observations)

        log_dict.update(
            dict(
                log_pi=log_pi.mean().item(),
                policy_loss=policy_loss.item(),
                alpha_loss=alpha_loss.item(),
                alpha=alpha.item(),
                std=std.mean().item(),
                actions_div=np.sqrt(F.mse_loss(actions, new_actions).item() / actions.shape[-1]),
            )
        )

        """ Q function loss """
        qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
            observations, actions, next_observations, rewards, dones, alpha, log_dict, trusts, online_steps
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if online_steps == 0:
            data_log_pi = self.actor.log_prob(observations, actions)
            base_log_prob = self.actor.log_prob_away_from_mean(observations, self.n_away)
            lowest_log_prob = self.actor.log_prob_away_from_mean(observations, 10)
            data_log_pi = torch.clamp(data_log_pi, min=lowest_log_prob)

            beta_weight = torch.mean((data_log_pi - base_log_prob).detach(), dim=-1)
            beta = self.beta(observations) * trusts if self.select_actions else self.beta(observations)

            beta_loss = (beta_weight.detach() * beta).mean()

            beta_weight_log = beta_weight[trusts == 1] if self.select_actions else beta_weight

            if self.update_threshold and self.total_it % self.update_period == 0:
                if beta_weight_log.mean() > 0:
                    self.update_threshold = False
                if self.update_threshold:
                    self.n_away = min(self.max_n_away, self.n_away + self.increase_step)

            log_beta = self.beta(observations)

            log_dict.update(
                dict(
                    beta_weight_min=beta_weight.min().item(),
                    beta_weight_mean=beta_weight.mean().item(),
                    beta_weight_max=beta_weight.max().item(),
                    beta_weight=beta_weight.mean().item(),
                    beta_min=log_beta.min().item(),
                    beta_mean=log_beta.mean().item(),
                    beta_max=log_beta.max().item(),
                    constrained_nums=beta_weight[beta_weight < 0].shape[0],
                    base_log_prob=base_log_prob.mean().item(),
                    data_log_pi=data_log_pi.mean().item(),
                    n_away=self.n_away,
                    beta_loss=beta_loss.item()
                )
            )
            self.beta_optimizer.zero_grad()
            beta_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.beta.parameters(), 0.5)
            self.beta_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
            "beta": self.beta,
            "beta_optim": self.beta_optimizer
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
        self.total_it = state_dict["total_it"]

        self.beta = state_dict["beta"]
        self.beta_optimizer = state_dict["beta_optim"]


@pyrallis.wrap()
def train(config: TrainConfig):
    config = load_train_config_auto(config)
    env = gym.make(config.env)

    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    max_steps = env._max_episode_steps
    ref_min_score = env.ref_min_score

    dataset = d4rl.qlearning_dataset(env)

    lim = 1e-4
    dataset['actions'] = np.clip(dataset['actions'], -1 + lim, 1 - lim)

    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(
            dataset,
            config.env,
            reward_scale=config.reward_scale,
            reward_bias=config.reward_bias,
        )

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    critic_1 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_2 = FullyConnectedQFunction(state_dim, action_dim, config.orthogonal_init).to(
        config.device
    )
    critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), config.qf_lr)
    critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), config.qf_lr)

    actor = TanhGaussianPolicy(
        state_dim,
        action_dim,
        max_action,
        log_std_multiplier=config.policy_log_std_multiplier,
        orthogonal_init=config.orthogonal_init,
    ).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)

    kwargs = {
        "env": config.env,
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "target_entropy": -np.prod(env.action_space.shape).item(),
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
        # Adaptive coefficient
        "min_n_away": config.min_n_away,
        "max_n_away": config.max_n_away,
        "update_period": config.update_period,
        "beta_lr": config.beta_lr,
        "decay_steps": config.decay_steps,
        "select_actions": config.select_actions,
        "loosen_bound": config.loosen_bound
    }
    print("---------------------------------------")
    print(f"Training CQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ContinuousCQL(**kwargs)

    writer = SummaryWriter(os.path.join('log/cql', config.env, str(config.seed)), write_to_disk=True)

    # pre-select stata-action pairs with high-value
    if config.select_actions:
        if config.select_method == 'iql':
            qv_learner_dir = os.path.join('model/iql_qv', config.env, str(config.seed))
            if not os.path.exists(qv_learner_dir):
                os.makedirs(qv_learner_dir)
            qv_learner_model = qv_learner_dir + f'/{config.iql_tau}_model.pth'
            IQL_qv_learner = IQL_qv(state_dim, action_dim, config.buffer_size, max_action, config.iql_tau,
                                    device=config.device, dataset=dataset)
            if os.path.exists(qv_learner_model):
                IQL_qv_learner.load_state_dict(torch.load(qv_learner_model))
                replay_buffer = IQL_qv_learner.modify_trusts()
            else:
                replay_buffer = IQL_qv_learner.train(writer, qv_learner_model)
        else:
            replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
            replay_buffer.load_d4rl_dataset(dataset)
            if 'antmaze' in config.env:
                select_trustable_trajectories(dataset, replay_buffer, ref_min_score, max_steps,
                                            reward_scale=config.reward_scale, reward_bias=config.reward_bias)
            else:
                traj_selection_for_dense_rewards(dataset, replay_buffer, max_steps, ref_return=config.ref_return)
    else:
        replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
        replay_buffer.load_d4rl_dataset(dataset)

    state, done = env.reset(), False
    episode_return = 0
    episode_step = 0
    goal_achieved = False

    eval_successes = []
    train_successes = []
    offline_evaluations = []
    online_evaluations = []

    online_replay_buffer = Online_ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    online_replay_buffer.initial(symbol='all', replay_buffer=replay_buffer)

    model_save_dir = os.path.join('./model/cql', config.env, str(config.seed))

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir, exist_ok=True)

    if config.load_offline:
        offline_model = torch.load(model_save_dir + '/offline_model.pth')
        trainer.load_state_dict(offline_model)
        config.offline_timesteps = 0
        print('Offline model has been loaded')

    for t in range(int(config.offline_timesteps) + int(config.online_timesteps)):
        if t == config.offline_timesteps:
            torch.save(trainer.state_dict(), model_save_dir + '/offline_model.pth')
            print("Offline model has been saved, start online tuning")
            online_replay_buffer.empty()
            online_replay_buffer.initial(symbol=config.load_way, replay_buffer=replay_buffer)
        if t >= config.offline_timesteps:
            episode_step += 1
            action = actor.act(np.array(state), config.device)
            next_state, reward, done, env_infos = env.step(action)
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)

            episode_return += reward
            real_done = False  # Episode can timeout which is different from done
            if done and episode_step < max_steps:
                real_done = True

            if config.normalize_reward:
                reward = modify_reward_online(
                    reward,
                    config.env,
                    reward_scale=config.reward_scale,
                    reward_bias=config.reward_bias,
                    **reward_mod_dict,
                )
            online_replay_buffer.add_transition(state, action, reward, next_state, real_done)
            state = next_state

            if done:
                state, done = env.reset(), False
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                if is_env_with_goal:
                    train_successes.append(goal_achieved)
                    writer.add_scalar('train_regret', np.mean(1 - np.array(train_successes)),
                                      t - config.offline_timesteps)
                    writer.add_scalar('is_success', float(goal_achieved), t - config.offline_timesteps)
                writer.add_scalar('episode_return', episode_return, t - config.offline_timesteps)
                normalized_score = env.get_normalized_score(episode_return) * 100.0
                writer.add_scalar('online_normalized_episode_score', normalized_score, t - config.offline_timesteps)
                writer.add_scalar('episode_length', episode_step, t - config.offline_timesteps)
                episode_return = 0
                episode_step = 0
                goal_achieved = False

        if t < config.offline_timesteps or t >= config.offline_timesteps + config.warmup_steps:
            time_steps = t - config.offline_timesteps - config.warmup_steps
            batch = online_replay_buffer.sample(config.batch_size)
            log_dict = trainer.train(batch, max(time_steps + 1, 0))

            # Evaluate episode
            if (t + 1) % config.eval_freq == 0:
                print(f"Time steps: {t + 1}")
                eval_scores, success_rate = eval_actor(
                    env,
                    actor,
                    device=config.device,
                    n_episodes=config.n_episodes,
                    seed=config.seed,
                )
                eval_score = eval_scores.mean()
                normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
                print("---------------------------------------")
                print(
                    f"Evaluation over {config.n_episodes} episodes: "
                    f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
                )
                print("---------------------------------------")

                if t < config.offline_timesteps:
                    writer.add_scalar('offline_normalized_eval_score', normalized_eval_score, t)
                    writer.add_scalar('offline_alpha', trainer.log_alpha().exp(), t)
                    writer.add_scalar('offline_std', log_dict['std'], t)
                    writer.add_scalar('offline_qf1', log_dict['average_qf1'], t)
                    writer.add_scalar('beta_min', log_dict['beta_min'], t)
                    writer.add_scalar('beta_mean', log_dict['beta_mean'], t)
                    writer.add_scalar('beta_max', log_dict['beta_max'], t)
                    writer.add_scalar('beta_weight', log_dict['beta_weight'], t)
                    writer.add_scalar('constrained_nums', log_dict['constrained_nums'], t)
                    writer.add_scalar('base_log_prob', log_dict['base_log_prob'], t)
                    writer.add_scalar('data_log_pi', log_dict['data_log_pi'], t)
                    writer.add_scalar('n_away', log_dict['n_away'], t)
                    writer.add_scalar('qf1_loss', log_dict['qf1_loss'], t)
                    writer.add_scalar('cql_min_qf1_loss', log_dict['cql_min_qf1_loss'], t)
                    writer.add_scalar('beta_loss', log_dict['beta_loss'], t)
                    offline_evaluations.append(normalized_eval_score)

                else:
                    online_log_steps = t - config.offline_timesteps
                    writer.add_scalar('online_normalized_eval_score', normalized_eval_score, online_log_steps)
                    writer.add_scalar('online_alpha', trainer.log_alpha().exp(), online_log_steps)
                    writer.add_scalar('online_std', log_dict['std'], online_log_steps)
                    writer.add_scalar('online_qf1', log_dict['average_qf1'], online_log_steps)
                    online_evaluations.append(normalized_eval_score)
                    if is_env_with_goal:
                        eval_successes.append(success_rate)
                        writer.add_scalar('eval_regret', np.mean(1 - np.array(eval_successes)), online_log_steps)
                        writer.add_scalar('success_rate', success_rate, online_log_steps)

    if config.online_timesteps > 0:
        torch.save(trainer.state_dict(), model_save_dir + '/online_model.pth')
        print("Online tuning ends and online model has been saved")
    else:
        torch.save(trainer.state_dict(), model_save_dir + '/offline_model.pth')
        print("Offline model has been saved")

    if len(offline_evaluations) > 1:
        offline_steps_list = [i * config.eval_freq for i in list(range(1, len(offline_evaluations) + 1))]
        offline_data_dict = {}
        offline_data_dict['steps'] = offline_steps_list
        offline_data_dict['offline_eval_returns'] = offline_evaluations

        offline_file_path = model_save_dir + '/offline_data.json'
        with open(offline_file_path, "w") as offline_json_file:
            json.dump(offline_data_dict, offline_json_file, indent=2)

    if len(online_evaluations) > 1:
        online_steps_list = [i * config.eval_freq for i in list(range(1, len(online_evaluations) + 1))]
        online_data_dict = {}
        online_data_dict['steps'] = online_steps_list
        online_data_dict['online_eval_returns'] = online_evaluations

        online_file_path = model_save_dir + '/online_data.json'
        with open(online_file_path, "w") as online_json_file:
            json.dump(online_data_dict, online_json_file, indent=2)



if __name__ == "__main__":
    train()
