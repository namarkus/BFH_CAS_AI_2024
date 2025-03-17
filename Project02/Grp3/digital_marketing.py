# -*- coding: utf-8 -*-
"""
Digital Marketing Simulation with TorchRL
=========================================
Dieses Skript simuliert die Anwendung von Reinforcement Learning (RL) auf das digitale Marketing.
Es verwendet die Bibliothek TorchRL, die auf PyTorch basiert. Die Umgebung wird durch die Klasse
DigitalMarketingEnv definiert, die die Schnittstelle für TorchRL implementiert.
"""
import time
import torch.optim as optim
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import (LazyTensorStorage, ReplayBuffer)
from torchrl.envs import (StepCounter, TransformedEnv)
from torchrl.modules import MLP, EGreedyModule, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate
import _logging as log
import _metrics as metrics
from _digital_marketing_env import DigitalMarketingEnv

REQUESTED_EPISODES = 10

def create_env():
  global base_env
  return TransformedEnv(base_env, StepCounter(max_steps=base_env.get_maximum_samples() * REQUESTED_EPISODES))

if __name__ == '__main__':
    log.start_logger("DigitalMarketing", "prd")
    log.app_logger().info("Starte die Simulation des digitalen Marketings mit TorchRl...")
    metrics = metrics.TensorBoardMonitor.instance()
    base_env = DigitalMarketingEnv()
    log.app_logger().info(f"Umgebung wurde erstellt. Sie stellt {base_env.get_maximum_samples()} Samples pro Episode (= Marketingkampagne) zur Verfügung.")
    value_mlp = MLP(in_features=base_env.num_features, out_features=base_env.action_spec.shape[-1], num_cells=[64, 128, 128])
    value_net = TensorDictModule(value_mlp, in_keys=["observation"], out_keys=["action_value"])
    policy = TensorDictSequential(value_net, QValueModule(spec=base_env.action_spec))
    exploration_module = EGreedyModule(
        base_env.action_spec, annealing_num_steps=base_env.get_maximum_samples() / 2, eps_init=0.9
    )
    policy_explore = TensorDictSequential(policy, exploration_module)

    init_rand_steps = 200
    collector = SyncDataCollector(
        create_env_fn = create_env,
        policy = policy_explore,
        frames_per_batch=1,  # ✅ Match the batch size
        total_frames=(base_env.get_maximum_samples()  * REQUESTED_EPISODES),
        init_random_frames=init_rand_steps,
        storing_device="cpu",  # ✅ Ensure storing happens on CPU
        split_trajs=False,  # ✅ Prevent trajectory splitting from dropping keys
        exploration_type="mode"
    )
    rb = ReplayBuffer(storage=LazyTensorStorage(100_000))
    loss = DQNLoss(value_network=policy, action_space=base_env.action_spec, delay_value=True)
    optim = Adam(loss.parameters(), lr=0.02)
    updater = SoftUpdate(loss, eps=0.99)
    total_count = 0
    total_episodes = 0
    solved_episodes = 0
    lost_episodes = 0
    optim_steps = 1 # ???
    t0 = time.time()
    for i, data in enumerate(collector):
        rb.extend(data) # Daten in Replay Buffer ergänzen
        max_length = rb[:]["next", "step_count"].max()
        if len(rb) > init_rand_steps:
            for j in range(optim_steps):
                sample = rb.sample(base_env.keyword_count) # alle Keywords in einem Batch verarbeiten.
                loss_vals = loss(sample)
                log.app_logger().debug("loss function is called:", loss_vals)
                loss_vals["loss"].backward()
                metrics.log_metrics({
                    "1 - Loss": loss_vals["loss"],
                    "1 - Reward": data["next", "reward"].sum()
                    }, step=total_count)
                optim.step()
                optim.zero_grad()
                exploration_module.step(data.numel()) # Update exploration factor
                updater.step() # Update target params
                total_count += data.numel()
                if data["next", "done"]:
                    reward = data["next", "reward"].sum()
                    log.app_logger().info(f"Episode {total_episodes} beendet mit Reward {reward}.")
                    metrics.log_metrics({"2 - Episode Success": reward}, step=total_episodes)
                    if reward > 0:
                        solved_episodes += 1
                    else:
                        lost_episodes += 1
                #print(f'Reward: {data["next", "reward"]} - Loss: {loss_vals["loss"]} - Epsisode: {data["next", "done"]}')
                total_episodes += data["next", "done"].sum()
    t1 = time.time()
    base_env.stop()
    log.app_logger().info(f"Ergebnis:  Von {total_episodes} Episoden wurden {solved_episodes} gelöst und {lost_episodes} verloren.")
    log.app_logger().info(f"Simulation des digitalen Marketings mit TorchRl nach {t1-t0}s beendet.")

