#!/usr/bin/env python3
import argparse
import gymnasium as gym

from lib import model

import numpy as np
import torch

ENVIRONMENT_ID = "LunarLanderContinuous-v2"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=False, default="saves/LunarLanderContinuous-v2_a2c_1env_lr0.0001_bs32_4rs_1e-05eps/best_+151.903_2840000.dat", help="Model file to load")
    parser.add_argument("-r", "--record", default="recordings" ,help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    env = gym.make(ENVIRONMENT_ID, render_mode='rgb_array')
    if args.record is not None:
        env = gym.wrappers.RecordVideo(env, video_folder=args.record)

    net = model.ModelA2C(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu'), weights_only=True))

    obs, _ = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor(np.expand_dims(obs, 0))
        mu_v, var_v, val_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        obs, reward, done, is_tr, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done or is_tr:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
    env.close()
