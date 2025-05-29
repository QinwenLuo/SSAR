import os
import random
from typing import Dict, Optional, Tuple, Union
import gym
import numpy as np
import torch
import torch.nn as nn
from gym.envs.classic_control.acrobot import bound


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
        env: gym.Env,
        state_mean: Union[np.ndarray, float] = 0.0,
        state_std: Union[np.ndarray, float] = 1.0,
        reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
                       state - state_mean
               ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward


@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    successes = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        goal_achieved = False
        while not done:
            action = actor.act(state, device)
            state, reward, done, env_infos = env.step(action)
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards), np.mean(successes)


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def select_trustable_trajectories(dataset, replay_buffer, ref_min_score, max_steps,
                                  reward_scale: float=1.0, reward_bias: float=-1.0):
    ep_len = 0
    N = len(dataset["rewards"])
    trusts = np.zeros_like(dataset["rewards"])
    for t, (r, d) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        ep_len += 1
        is_last_step = ((t == N - 1) or (np.linalg.norm(dataset["observations"][t + 1] - dataset["next_observations"][
            t]) > 1e-6) or ep_len == max_steps)

        if d or is_last_step:
            if r != ref_min_score * reward_scale + reward_bias:
                trusts[t + 1 - ep_len:t + 1] = 1
            ep_len = 0
    replay_buffer.modify_trusts(np.where(trusts == 0)[0])

def traj_selection_for_dense_rewards(dataset, replay_buffer, max_steps, ref_return):
    ep_len = 0
    N = len(dataset["rewards"])
    trusts = np.zeros_like(dataset["rewards"])
    traj_return = 0
    for t, (r, d) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        ep_len += 1
        traj_return += r
        is_last_step = ((t == N - 1) or (np.linalg.norm(dataset["observations"][t + 1] - dataset["next_observations"][
            t]) > 1e-6) or ep_len == max_steps)

        if d or is_last_step:
            if traj_return > ref_return:
                trusts[t + 1 - ep_len:t + 1] = 1
            ep_len = 0
            traj_return = 0

    replay_buffer.modify_trusts(np.where(trusts == 0)[0])


def avg_high_returns(dataset, max_steps, ref_return=None):
    ep_len = 0
    N = len(dataset["rewards"])
    trusts = np.zeros_like(dataset["rewards"])
    returns_list = []
    traj_return = 0
    for t, (r, d) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        ep_len += 1
        traj_return += r
        is_last_step = ((t == N - 1) or (np.linalg.norm(dataset["observations"][t + 1] - dataset["next_observations"][
            t]) > 1e-6) or ep_len == max_steps)

        if d or is_last_step:
            if ref_return is not None:
                if traj_return > ref_return:
                    trusts[t + 1 - ep_len:t + 1] = 1
                    returns_list.append(traj_return)
            else:
                returns_list.append(traj_return)
            ep_len = 0
            traj_return = 0

    if ref_return is None:
        returns_list = np.sort(np.array(returns_list))[-10:]
    avg_high_return = np.array(returns_list).mean()
    return avg_high_return