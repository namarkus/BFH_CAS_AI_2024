#!/usr/bin/env python3
import gymnasium as gym
import ptan
from ptan.experience import VectorExperienceSourceFirstLast
from ptan.common.utils import TBMeanTracker
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

import sys
import time
import typing as tt

import torch.nn as nn
from ptan.experience import ExperienceFirstLast
import optuna
from optuna.trial import TrialState


FASTEST_DEVICE = {
    torch.cuda.is_available(): "cuda", # GPU
    torch.backends.mps.is_available(): "mps" # Metal-Performance-Service, z.B. Mac Mx;
}.get(True, "cpu")

ENVIRONEMNT_ID = "LunarLander-v2"
EXPECTED_MEDIAN_REWARD = 175

class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward
        self.current_mean_reward = 0.0

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        self.current_mean_reward = mean_reward
        if frame % 1000 == 0:
            epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
            print(f"Idx: {frame:,}: {len(self.total_rewards):,} rewards, {mean_reward:.2f} current mean reward{epsilon_str}")
            sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d tries!" % frame)
            return True
        if frame > 1_000_000 and mean_reward < -500:
            print("Not solved after 1M tries")
            raise optuna.TrialPruned()
        if frame > 2_000_000 and mean_reward < -200:
            print("Not solved after 2M tries")
            raise optuna.TrialPruned()
        if frame > 4_000_000 and mean_reward < 0:
            print("Not solved after 4M tries")
            raise optuna.TrialPruned()
        if frame > 10_000_000:
            print("Not solved after 10M tries")
            raise optuna.TrialPruned()
        return False


class A2CNet(nn.Module):
    def __init__(self, input_size: int, n_actions: int, number_of_nodes = 128, interface_size=128, number_of_subnodes=512):
        super(A2CNet, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, number_of_nodes),
            nn.ReLU(),
            nn.Linear(number_of_nodes, interface_size)
        )

        self.policy = nn.Sequential(
            nn.Linear(interface_size, number_of_subnodes),
            nn.ReLU(),
            nn.Linear(number_of_subnodes, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(interface_size, number_of_subnodes),
            nn.ReLU(),
            nn.Linear(number_of_subnodes, 1)
        )

    def forward(self, x: torch.ByteTensor) -> tt.Tuple[torch.Tensor, torch.Tensor]:
        shared_out = self.shared(x)
        return self.policy(shared_out), self.value(shared_out)



def unpack_batch(batch: tt.List[ExperienceFirstLast], net: A2CNet,
                 device: torch.device, gamma: float, reward_steps: int):
    """
    Convert batch into training tensors
    :param batch: batch to process
    :param net: network to useß
    :param gamma: gamma value
    :param reward_steps: steps of reward
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.asarray(exp.state))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.asarray(exp.last_state))

    states_t = torch.FloatTensor(np.asarray(states)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_t = torch.FloatTensor(np.asarray(last_states)).to(device)
        last_vals_t = net(last_states_t)[1]
        last_vals_np = last_vals_t.data.cpu().numpy()[:, 0]
        last_vals_np *= gamma ** reward_steps
        rewards_np[not_done_idx] += last_vals_np
    ref_vals_t = torch.FloatTensor(rewards_np).to(device)
    return states_t, actions_t, ref_vals_t

def objective(trial: optuna.trial.Trial) -> float:
    #learning_rate = trial.suggest_categorical("learning_rate", [0.01, 0.001, 0.0001])
    learning_rate = trial.suggest_categorical("learning_rate", [0.001])
    #batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [128])
    #environment_count = trial.suggest_categorical("num_envs", [25, 50, 100])
    environment_count = trial.suggest_categorical("num_envs", [20, 25])
    #reward_steps = trial.suggest_categorical("reward_steps", [1, 4, 8,16])
    reward_steps = trial.suggest_categorical("reward_steps", [4])
    #number_of_nodes = trial.suggest_categorical("number_of_nodes", [128, 256, 512])
    number_of_nodes = trial.suggest_categorical("number_of_nodes", [128, 256])
    #interface_size = trial.suggest_categorical("interface_size", [128, 256, 512, 1024])
    interface_size = trial.suggest_categorical("interface_size", [256, 512])
    #number_of_subnodes = trial.suggest_categorical("number_of_subnodes", [256, 512, 1024, 2048])
    number_of_subnodes = trial.suggest_categorical("number_of_subnodes", [512, 1024])
    CLIP_GRAD = 0.1
    ENTROPY_BETA = 0.01
    GAMMA = 0.99

    LOG_LABEL = f"{ENVIRONEMNT_ID}-A2CNet{number_of_nodes}x{interface_size}x{number_of_subnodes}-{environment_count}envs_lr{learning_rate}_bs{batch_size}_rs{reward_steps}"
    env_factories = [
        lambda: gym.make(ENVIRONEMNT_ID, render_mode="rgb_array")
        for _ in range(environment_count)
    ]
    env = gym.vector.SyncVectorEnv(env_factories)
    writer = SummaryWriter(comment=LOG_LABEL)
    print(LOG_LABEL)
    net = A2CNet(env.envs[0].observation_space.shape[0], env.envs[0].action_space.n, number_of_nodes = number_of_nodes, interface_size=interface_size, number_of_subnodes=number_of_subnodes).to(FASTEST_DEVICE)
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=FASTEST_DEVICE)
    exp_source = VectorExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=reward_steps)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, eps=1e-3)
    batch = []
    with RewardTracker(writer, stop_reward=EXPECTED_MEDIAN_REWARD) as tracker:
        with TBMeanTracker(writer, batch_size=batch_size) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)
                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break
                if len(batch) < batch_size:
                    continue
                states_t, actions_t, vals_ref_t = unpack_batch(batch, net, device=FASTEST_DEVICE, gamma=GAMMA, reward_steps=reward_steps)
                batch.clear()

                optimizer.zero_grad()
                logits_t, value_t = net(states_t)
                loss_value_t = F.mse_loss(value_t.squeeze(-1), vals_ref_t)

                log_prob_t = F.log_softmax(logits_t, dim=1)
                adv_t = vals_ref_t - value_t.detach()
                log_act_t = log_prob_t[range(batch_size), actions_t]
                log_prob_actions_t = adv_t * log_act_t
                loss_policy_t = -log_prob_actions_t.mean()

                prob_t = F.softmax(logits_t, dim=1)
                entropy_loss_t = ENTROPY_BETA * (prob_t * log_prob_t).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_t.backward(retain_graph=True)
                grads = np.concatenate([
                    p.grad.data.cpu().numpy().flatten()
                    for p in net.parameters() if p.grad is not None
                ])

                # apply entropy and value gradients
                loss_v = entropy_loss_t + loss_value_t
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                # get full loss
                loss_v += loss_policy_t

                tb_tracker.track("advantage", adv_t, step_idx)
                tb_tracker.track("values", value_t, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_t, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_t, step_idx)
                tb_tracker.track("loss_policy", loss_policy_t, step_idx)
                tb_tracker.track("loss_value", loss_value_t, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max", np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var", np.var(grads), step_idx)
                #frame = env.envs[0].render()  # Frame im Format (H, W, 3)
                #frame = np.transpose(frame, (2, 0, 1))  # Von (H, W, C) nach (C, H, W).
                #tb_tracker.writer.add_image("Screenshot", frame, step_idx)
    return tracker.total_rewards[-1:]

if __name__ == "__main__":
    study = optuna.create_study(study_name=f"{ENVIRONEMNT_ID}.{EXPECTED_MEDIAN_REWARD}", direction="maximize")
    #study.optimize(objective, n_trials=25)
    study.optimize(objective, n_trials=5)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
      print("    {}: {}".format(key, value))    