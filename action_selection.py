import copy
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from utils import soft_update
from dataset_utils import ReplayBuffer

TensorBatch = List[torch.Tensor]


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
            self,
            dims,
            activation_fn: Callable[[], nn.Module] = nn.ReLU,
            output_activation_fn: Callable[[], nn.Module] = None,
            squeeze_output: bool = False,
            dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwinQ(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class IQL_qv:
    def __init__(
            self,
            state_dim,
            action_dim,
            buffer_size,
            max_action: float,
            iql_tau: float = 0.7,
            discount: float = 0.99,
            tau: float = 0.005,
            device: str = "cpu",
            dataset=None,
            batch_size=256,
            training_times=int(1e6),
            log_freq=int(5e3)
    ):
        self.max_action = max_action
        self.qf = TwinQ(state_dim, action_dim).to(device)
        self.vf = ValueFunction(state_dim).to(device)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=3e-4)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=3e-4)
        self.iql_tau = iql_tau
        self.discount = discount
        self.tau = tau

        self.device = device

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size, device)

        self.replay_buffer.load_d4rl_dataset(dataset)

        self.batch_size = batch_size
        self.training_times = training_times
        self.log_freq = log_freq

    def _update_v(self, observations, actions, log_dict):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["iql_value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

    def _update_q(
            self,
            next_v: torch.Tensor,
            observations: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            terminals: torch.Tensor,
            log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["iql_q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        soft_update(self.q_target, self.qf, self.tau)

    def train(self, writer, save_model_model):
        for t in tqdm.tqdm(range(self.training_times), desc='trustable actions seeking ......'):
            (observations, actions, rewards, next_observations, dones, _) = self.replay_buffer.sample(self.batch_size)
            log_dict = {}

            with torch.no_grad():
                next_v = self.vf(next_observations)

            self._update_v(observations, actions, log_dict)
            rewards = rewards.squeeze(dim=-1)
            dones = dones.squeeze(dim=-1)

            self._update_q(next_v, observations, actions, rewards, dones, log_dict)

            if (t + 1) % self.log_freq == 0:
                for k, v in log_dict.items():
                    writer.add_scalar(f'{k}', v, t)

        torch.save(self.state_dict(), save_model_model)

        return self.modify_trusts()

    def modify_trusts(self):
        states, actions = self.replay_buffer.get_sa()
        total_size = states.shape[0]
        for i in range(total_size // self.batch_size + 1):
            batch_states = states[i * self.batch_size:min((i + 1) * self.batch_size, total_size)]
            batch_actions = actions[i * self.batch_size:min((i + 1) * self.batch_size, total_size)]
            q = self.qf(batch_states, batch_actions)
            v = self.vf(batch_states)
            indx = torch.where(q < v)[0] + i * self.batch_size
            self.replay_buffer.modify_trusts(indx)

        trust_nums = self.replay_buffer._size - torch.where(self.replay_buffer._trusts==0)[0].shape[0]

        return self.replay_buffer

    def state_dict(self):
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])
