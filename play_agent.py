"""Play Space Invaders with a trained agent (no training needed).

Usage:
    python play_agent.py                          # Double DQN, saves gameplay.mp4
    python play_agent.py --model vanilla          # Vanilla DQN instead
    python play_agent.py --episodes 3 --render    # also open a live window
"""
import argparse

import ale_py
import gymnasium as gym
import imageio
import keras
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)

MODELS = {"ddqn": "models/double_dqn.keras", "vanilla": "models/vanilla_dqn.keras"}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["ddqn", "vanilla"], default="ddqn")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--epsilon", type=float, default=0.05, help="small randomness so the agent doesn't get stuck")
parser.add_argument("--render", action="store_true", help="show a live window")
parser.add_argument("--out", default="gameplay.mp4", help="video of the first episode")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

def build_model(num_actions=6):
    """Same network as the notebook (Mnih et al., 2015)."""
    return keras.Sequential([
        keras.Input(shape=(4, 84, 84)),
        keras.layers.Lambda(lambda t: keras.ops.transpose(t, [0, 2, 3, 1]), output_shape=(84, 84, 4)),
        keras.layers.Rescaling(1.0 / 255.0),
        keras.layers.Conv2D(32, 8, strides=4, activation="relu"),
        keras.layers.Conv2D(64, 4, strides=2, activation="relu"),
        keras.layers.Conv2D(64, 3, strides=1, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(num_actions, activation="linear"),
    ])


# Rebuild the network and load only the saved weights. This works on any Python version,
# unlike load_model(), which can crash on the Lambda layer if Python versions differ.
model = build_model()
model.load_weights(MODELS[args.model])

# Same environment settings as training
env = gym.make("ALE/SpaceInvaders-v5", frameskip=1, repeat_action_probability=0.0,
               full_action_space=False, render_mode="human" if args.render else "rgb_array")
env = AtariPreprocessing(env, frame_skip=4)
env = FrameStackObservation(env, 4)
rng = np.random.default_rng(args.seed)

scores, frames = [], []
for ep in range(args.episodes):
    obs, _ = env.reset(seed=args.seed + ep)
    state, total, done = np.array(obs), 0.0, False
    while not done:
        if ep == 0 and not args.render:
            frames.append(env.render())
        if rng.random() < args.epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(model(state[None], training=False)[0]))
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        state = np.array(obs)
        done = terminated or truncated
    scores.append(total)
    print(f"Episode {ep + 1}: score {total:.0f}")

print(f"Mean score over {args.episodes} episode(s): {np.mean(scores):.1f}")
if frames:
    imageio.mimsave(args.out, frames, fps=30, macro_block_size=1)
    print("Saved", args.out)
env.close()
