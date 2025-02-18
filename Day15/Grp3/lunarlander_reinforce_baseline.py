#!/usr/bin/env python3
import gymnasium as gym
import ptan
import numpy as np
import typing as tt
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import RecordVideo

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4

class PGN(nn.Module):
    def __init__(self, input_size: int, n_actions: int):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def calc_qvals(rewards: tt.List[float]) -> tt.List[float]:
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    res = list(reversed(res))
    mean_q = np.mean(res)
    return [q - mean_q for q in res]

def episode_trigger(episode_id):
    return episode_id % 1000 == 0 or mean_rewards > 175

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="Day15/Grp3/videos", episode_trigger=episode_trigger)
    writer = SummaryWriter(logdir="Day15/Grp3/runs", comment="-lunarlander-reinforce-baseline")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                   apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_idx = 0
    done_episodes = 0
    mean_rewards = 0.0

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_states, cur_actions, cur_rewards = [], [], []

    for step_idx, exp in enumerate(exp_source):
        cur_states.append(exp.state)
        cur_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        if exp.last_state is None:
            batch_states.extend(cur_states)
            batch_actions.extend(cur_actions)
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_states.clear()
            cur_actions.clear()
            cur_rewards.clear()
            batch_episodes += 1

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            if done_episodes % 100 == 0:
                print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                    step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 175:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        states_v = torch.as_tensor(np.asarray(batch_states))
        batch_actions_t = torch.as_tensor(batch_actions)
        batch_qvals_v = torch.as_tensor(batch_qvals)

        optimizer.zero_grad()
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    writer.close()
