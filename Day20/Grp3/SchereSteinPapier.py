# -*- coding: utf-8 -*-

# Imports
from typing import Optional

import pandas as pd
import numpy as np

import torch
from torch.optim import Adam

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from torchrl.modules import EGreedyModule, MLP, QValueModule
from torchrl.data import OneHot, Composite, Unbounded
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.envs import EnvBase, StepCounter, TransformedEnv
from torchrl.envs.utils import check_env_specs
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Aktionen und Rewards definieren
ACTIONS = {0: "Schere", 1: "Stein", 2: "Papier"}
REWARDS = {
    (0, 0): 0, (0, 1): -1, (0, 2): 1,
    (1, 0): 1, (1, 1): 0, (1, 2): -1,
    (2, 0): -1, (2, 1): 1, (2, 2): 0
}
RANDOM = np.random.default_rng(42)
def generate_random_moves(num_samples=10000):
    data = {
        # Unser Gegner hat leichten Hang zur Schere ;-)
        "move": RANDOM.choice([0, 1, 2], size=num_samples, p=[0.4, 0.3, 0.3])
    }
    return pd.DataFrame(data)

# Umgebung und ihr Verhalten definieren
class RockPaperScissorsEnv(EnvBase):
    def __init__(self, device="cpu"):
        super().__init__(device=device)
        self.dataset = generate_random_moves()
        self.num_features = 1
        self.observation_spec = Composite(observation=Unbounded(shape=(1,), dtype=torch.float32))
        self.action_spec = Composite(action=OneHot(n=len(ACTIONS.items()), dtype=torch.int64))
        self.reward_spec = Composite(reward=Unbounded(shape=(1,), dtype=torch.float32))

    def _step(self, tensordict):
        action = tensordict["action"].argmax(dim=-1).item()
        next_sample = self.dataset.sample(1).iloc[0]
        next_state = torch.tensor(
            [next_sample["move"]], dtype=torch.float32
        )
        reward = REWARDS[(action, next_sample["move"])]
        return TensorDict({
            "observation": next_state,
            "reward": torch.tensor([reward], dtype=torch.float32),
            "done": torch.tensor([False], dtype=torch.bool),
        }, batch_size=[])

    def _reset(self, tensordict=None):
        sample = self.dataset.sample(1).iloc[0]
        state = torch.tensor([sample["move"]], dtype=torch.float32)
        return TensorDict({
            "observation": state,
        }, batch_size=[])

    def _set_seed(self, seed: Optional[int]):
        pass

env = RockPaperScissorsEnv()
print("Observation Keys:", env.observation_keys)
print("Action Keys:", env.action_keys)
print("Reward Keys:", env.reward_keys)

value_mlp = MLP(in_features=env.num_features, out_features=env.action_spec.shape[-1], num_cells=[64, 64])
value_net = TensorDictModule(value_mlp, in_keys=["observation"], out_keys=["action_value"])
policy = TensorDictSequential(value_net, QValueModule(spec=env.action_spec))
exploration_module = EGreedyModule(
    env.action_spec, annealing_num_steps=10_000, eps_init=0.9
)
policy_explore = TensorDictSequential(policy, exploration_module)

def create_env():
  return TransformedEnv(RockPaperScissorsEnv(), StepCounter(max_steps=5_000))

init_rand_steps = 100
collector = SyncDataCollector(
    create_env_fn = create_env,
    policy = policy_explore,
    frames_per_batch=1,  # ✅ Match the batch size
    total_frames=5_000,
    init_random_frames=init_rand_steps,
    storing_device="cpu",  # ✅ Ensure storing happens on CPU
    split_trajs=False,  # ✅ Prevent trajectory splitting from dropping keys
    exploration_type="mode"
)
rb = ReplayBuffer(storage=LazyTensorStorage(10_000))

loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
optim = Adam(loss.parameters(), lr=0.02, weight_decay=1e-5)
updater = SoftUpdate(loss, eps=0.99)

total_count = 0
optim_steps = 5
batch_size = 100
wins = 0
draws = 0
losses = 0

for i, data in enumerate(collector):
    rb.extend(data)
    if len(rb) > init_rand_steps:
        for _ in range(optim_steps):
            sample = rb.sample(batch_size)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
        # Update exploration factor and target params after optimization
        exploration_module.step(data.numel())
        updater.step()

    # Update counters
    total_count += data.numel()
    rewards = data['next','reward'].view(-1).tolist()  # Extrahiere Rewards als Liste
    wins += rewards.count(1)  # Zähle Gewinne
    draws += rewards.count(0)  # Zähle Unentschieden
    losses += rewards.count(-1)  # Zähle Verluste
    if i % 100 == 0:
        # Log progress every 100 iterations
        print(
            f"Iteration {i}: Total steps: {total_count} - Wins: {wins}, Draws: {draws}, Losses: {losses} - Replay buffer size: {len(rb)}"
        )

print(
    f"Finished after {total_count} steps."
    f"Final Results - Wins: {wins}, Draws: {draws}, Losses: {losses}"
)