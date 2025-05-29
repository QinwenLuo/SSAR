import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import d4rl.gym_mujoco
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
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
    env: str = "walker2d-medium-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    offline_timesteps: int = int(1e6)  # Max time steps to run environment

    # TD3
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    # TD3 + BC
    alpha: float = 2.5
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward


    """adaptive constraint params"""
    iql_tau: float = 0.5  # the hyperparameter of IQL, set the same value of IQL
    select_actions: bool = True  # whether to select high-value actions
    select_method: str = 'traj'  # the method to obtain the sub-dataset, use 'traj' or 'iql'
    ref_return: int = 3000  # the return threshold for obtain the sub-dataset with high returns
    loosen_bound: bool = False  # whether to apply the regularization on the sub-dataset with high-value actions
    min_n_away: float = 1.0  # the initial threshold n_{start}
    max_n_away: float = 3.0  # the maximum threshold n_{end}
    update_period: int = 50000  # the interval of updating the threshold n
    beta_lr: float = 3e-8  # the learning rate of coefficient net

    """offline-to-online setting"""
    online_timesteps: int = 250000  # total online steps
    warmup_steps: int = 5000  # steps for warm up
    decay_steps: int = 400000  # the decay steps for the coefficients
    load_way: str = 'part'  # load all/part/none data, or half sample
    load_offline: bool = False  # use offline pre-trained model as initialization
    new_policy_freq: int = 2  # the interval of policy updates


def load_train_config_auto(config):
    env_name_lower = "_".join(config.env.split("-")[:1]).lower().replace("-", "_")
    env_lower = "_".join(config.env.split("-")[1:]).lower().replace("-", "_")

    file_path = os.path.join(f"config/td3_bc/{env_name_lower}", f"{env_lower}.yaml")

    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
        config_fields = fields(config)

        filtered_config_data = {field.name: config_data[field.name] for field in config_fields if
                                field.name in config_data}
        config = replace(config, **filtered_config_data)
        return config


def modify_reward(dataset: Dict, env_name: str, max_episode_steps: int = 1000) -> Dict:
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
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
        modification_data = {
            "reward_scale": 1.0,
            "reward_bias": -1.0
        }
    return modification_data


def modify_reward_online(reward, env_name, **kwargs):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    elif "antmaze" in env_name:
        reward = reward * kwargs["reward_scale"] + kwargs["reward_bias"]
    return reward


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)


class TD3_BC:
    def __init__(
            self,
            max_action: float,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            critic_1: nn.Module,
            critic_1_optimizer: torch.optim.Optimizer,
            critic_2: nn.Module,
            critic_2_optimizer: torch.optim.Optimizer,
            discount: float = 0.99,
            tau: float = 0.005,
            policy_noise: float = 0.2,
            noise_clip: float = 0.5,
            policy_freq: int = 2,
            expl_noise: float = 0.1,
            init_beta: float = 0.4,
            device: str = "cpu",
            min_n_away: float = 2.0,
            max_n_away: float = 3.0,
            update_period: int = 1,
            beta_lr: float = 5e-9,
            decay_steps: int = 250000,
            select_actions: bool = True,
            loosen_bound: bool = True
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.expl_noise = expl_noise

        self.total_it = 0

        # initialize the coefficient net
        self.beta = ParaNet(actor.net[0].in_features, init_beta, squeeze_output=False).to(device)
        self.beta_optimizer = torch.optim.Adam(self.beta.parameters(), lr=beta_lr)

        # initialize the threshold
        self.update_threshold = True
        self.n_away = min_n_away
        self.max_n_away = max_n_away
        self.increase_step = (self.max_n_away - self.n_away) / 10
        self.threshold = self.expl_noise ** 2 * self.n_away ** 2
        self.update_period = update_period

        self.select_actions = select_actions
        self.loosen_bound = loosen_bound

        self.decay_steps = decay_steps

    def decay_factor(self, online_steps):
        return 1 - online_steps / self.decay_steps

    def train(self, batch: TensorBatch, online_steps: int = 0) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, next_state, done, trusts = batch
        not_done = 1 - done

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()
        log_dict["q1"] = current_q1.mean().item()
        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            q = self.critic_1(state, pi)
            lmbda = 1 / q.abs().mean().detach()

            beta = self.beta(state) * trusts if self.loosen_bound else self.beta(state)

            if online_steps > 0 and self.decay_steps > 0:
                beta *= self.decay_factor(online_steps)

            action_mse = (pi - action) ** 2
            actor_loss = -lmbda * q.mean() + (beta.detach() * action_mse).mean()
            log_dict["actor_loss"] = actor_loss.item()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            if online_steps == 0:

                action_mse_max = torch.max(action_mse, dim=-1)[0].unsqueeze(-1).detach()

                action_mse_sum = action_mse.clamp(max=1.0).mean(-1, keepdims=True)

                beta_weight = self.threshold - action_mse_sum

                beta = self.beta(state) * trusts if self.select_actions else self.beta(state)

                beta_loss = (beta_weight.detach() * beta).mean()

                beta_weight_log = beta_weight[trusts == 1] if self.select_actions else beta_weight

                if self.update_threshold and self.total_it % self.update_period == 0:
                    if beta_weight_log.mean() > 0:
                        self.update_threshold = False
                    if self.update_threshold:
                        self.n_away = min(self.max_n_away, self.n_away + self.increase_step)
                        self.threshold = self.expl_noise ** 2 * self.n_away ** 2

                log_beta = beta[trusts == 1] if (self.loosen_bound and len(beta[trusts == 1]) > 0) else beta

                log_dict.update(
                    dict(
                        action_mse_max_mean=action_mse_max.mean(),
                        action_mse_mean=action_mse.mean(),
                        beta_min=log_beta.min(),
                        beta_mean=log_beta.mean(),
                        beta_max=log_beta.max(),
                        beta_weight=beta_weight.mean().item(),
                        constrained_nums=beta_weight[beta_weight < 0].shape[0],
                        n_away=self.n_away
                    )
                )

                self.beta_optimizer.zero_grad()
                beta_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.beta.parameters(), 0.5)
                self.beta_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
            "beta": self.beta,
            "beta_optim": self.beta_optimizer
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

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

    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(dataset, config.env)

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

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    critic_1 = Critic(state_dim, action_dim).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2 = Critic(state_dim, action_dim).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        "expl_noise": config.expl_noise,
        # TD3 + BC
        "init_beta": 1 / config.alpha,
        # Adaptive coefficient
        "min_n_away": config.min_n_away,
        "max_n_away": config.max_n_away,
        "update_period": config.update_period,
        "beta_lr": config.beta_lr,
        "decay_steps": config.decay_steps,
        "select_actions": config.select_actions,
        "loosen_bound": config.loosen_bound,
    }

    print("---------------------------------------")
    print(f"Training TD3 + BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = TD3_BC(**kwargs)

    writer = SummaryWriter(os.path.join('log/td3_bc', config.env, str(config.seed), str(config.select_actions),
                                        str(config.loosen_bound), str(config.select_method),
                                        str(config.ref_return)), write_to_disk=True)

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
                select_trustable_trajectories(dataset, replay_buffer, ref_min_score, max_steps)
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
        config.online_timesteps + 100,
        config.device,
    )
    online_replay_buffer.initial(symbol='all', replay_buffer=replay_buffer)

    model_save_dir = os.path.join('./model/td3_bc', config.env, str(config.seed))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir, exist_ok=True)

    if config.load_offline:
        offline_model = torch.load(model_save_dir + '/offline_model.pth')
        trainer.load_state_dict(offline_model)
        config.offline_timesteps = 0
        print('Offline model has been loaded')

    for t in range(int(config.offline_timesteps) + int(config.online_timesteps)):
        if t == 1000000:
            torch.save(trainer.state_dict(), model_save_dir + '/offline_model_1e6.pth')
        if t == config.offline_timesteps:
            trainer.policy_freq = config.new_policy_freq
            torch.save(trainer.state_dict(), model_save_dir + '/offline_model.pth')
            print("Offline model has been saved, start online tuning")
            online_replay_buffer.empty()
            online_replay_buffer.initial(symbol=config.load_way, replay_buffer=replay_buffer)

            if config.offline_timesteps == 0:
                print(f"Time steps: {t}")
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
                online_log_steps = t - config.offline_timesteps
                writer.add_scalar('online_normalized_eval_score', normalized_eval_score, online_log_steps)
                online_evaluations.append(normalized_eval_score)
                if is_env_with_goal:
                    eval_successes.append(success_rate)
                    writer.add_scalar('eval_regret', np.mean(1 - np.array(eval_successes)), online_log_steps)
                    writer.add_scalar('success_rate', success_rate, online_log_steps)

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
                reward = modify_reward_online(reward, config.env, **reward_mod_dict)
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
                    writer.add_scalar('beta_min', log_dict['beta_min'], t)
                    writer.add_scalar('beta_mean', log_dict['beta_mean'], t)
                    writer.add_scalar('beta_max', log_dict['beta_max'], t)
                    writer.add_scalar('beta_weight', log_dict['beta_weight'], t)
                    writer.add_scalar('constrained_nums', log_dict['constrained_nums'], t)
                    writer.add_scalar('action_mse_max_mean', log_dict['action_mse_max_mean'], t)
                    writer.add_scalar('action_mse_mean', log_dict['action_mse_mean'], t)
                    writer.add_scalar('offline_q1', log_dict['q1'], t)
                    writer.add_scalar('n_away', log_dict['n_away'], t)
                    offline_evaluations.append(normalized_eval_score)
                else:
                    online_log_steps = t - config.offline_timesteps
                    writer.add_scalar('online_normalized_eval_score', normalized_eval_score, online_log_steps)
                    writer.add_scalar('online_q1', log_dict['q1'], online_log_steps)
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
