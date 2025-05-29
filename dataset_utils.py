import torch
import numpy as np
from typing import Dict, List


TensorBatch = List[torch.Tensor]

class ReplayBuffer:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._trusts = torch.ones((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        trusts = self._trusts[indices]
        return [states, actions, rewards, next_states, dones, trusts]

    def add_transition(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def modify_trusts(self, indx):
        self._trusts[indx] = 0

    def replace_trusts(self, trusts):
        self._trusts[:self._pointer] = self._to_tensor(trusts.reshape(-1, 1))

    def get_sa(self):
        return self._states[:self._size], self._actions[:self._size]

    def filtrate_trajs(self, nums):
        indx_list, indx = [], []
        returns = []
        traj_return = 0
        for i in range(self._size - 1):
            traj_return += self._rewards[i].item()
            indx.append(i)
            if self._dones[i] == 1 or torch.norm(self._states[i + 1] - self._next_states[i]) > 1e-4:
                returns.append(traj_return)
                traj_return = 0
                indx_list.append(indx)
                indx = []

        returns = np.array(returns)
        sorted_indices = np.argsort(-returns)[:nums].tolist()
        flatten_indx = []
        for i in range(len(sorted_indices)):
            for j in range(len(indx_list[sorted_indices[i]])):
                flatten_indx.append(indx_list[sorted_indices[i]][j])
        self._size = len(flatten_indx)
        self._states[:self._size] = self._states[flatten_indx]
        self._actions[:self._size] = self._actions[flatten_indx]
        self._rewards[:self._size] = self._rewards[flatten_indx]
        self._next_states[:self._size] = self._next_states[flatten_indx]
        self._dones[:self._size] = self._dones[flatten_indx]
        self._trusts[:self._size] = self._trusts[flatten_indx]
        self._pointer = self._size


class Online_ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size, device):
        self.online_buffer = ReplayBuffer(state_dim, action_dim, buffer_size, device)
        self.offline_buffer = None
        self.symbol = 0
        self.params = [state_dim, action_dim, buffer_size, device]

    def initial(self, symbol: str, replay_buffer):
        if symbol == 'all':
            self.online_buffer = replay_buffer
        elif symbol == 'half':
            self.offline_buffer = replay_buffer
            self.symbol = 1
        elif symbol == 'part':
            self.online_buffer = replay_buffer
            self.online_buffer.filtrate_trajs(50)
        elif symbol == 'none':
            pass
        else:
            raise NameError

    def empty(self):
        self.online_buffer = ReplayBuffer(self.params[0], self.params[1], self.params[2], self.params[3])

    def add_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.online_buffer.add_transition(state, action, reward, next_state, done)

    def sample(self, batch_size: int):
        if self.symbol == 1:
            online_batch = self.online_buffer.sample(batch_size // 2)
            offline_batch = self.offline_buffer.sample(batch_size // 2)
            batch = merge_tensors(online_batch, offline_batch)
        else:
            batch = self.online_buffer.sample(batch_size)
        return batch

    def modify_trusts(self, data_nums):
        end_index = int(self.online_buffer._pointer)
        start_index = int(end_index - data_nums)
        self.online_buffer.modify_trusts(list(range(start_index, end_index)))

def merge_tensors(batch1, batch2):
    assert len(batch1) == len(batch2)
    batch = []
    for i in range(len(batch1)):
        batch.append(torch.cat([batch1[i], batch2[i]], dim=0))
    return batch
