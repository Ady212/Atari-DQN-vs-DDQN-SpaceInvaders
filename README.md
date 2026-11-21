# Deep Q-Learning on Atari Space Invaders: DQN vs Double DQN

CS6482 Deep Reinforcement Learning coursework (Assignment 2): a **Vanilla DQN** and a **Double DQN** agent trained to play **Space Invaders** (`ALE/SpaceInvaders-v5`) from raw pixels, using TensorFlow/Keras and Gymnasium.

## What's in the notebook

[`dqn_ddqn_space_invaders.ipynb`](dqn_ddqn_space_invaders.ipynb) covers:

1. **Why reinforcement learning** suits game playing.
2. **The environment:** observation and action spaces, and the preprocessing pipeline (grayscale, 84×84, frame skip 4, 4-frame stack).
3. **Vanilla DQN:** the Mnih et al. (2015) CNN, experience replay, a target network, ε-greedy exploration, reward clipping and the Huber loss.
4. **Evaluation:** episode reward, moving averages and the ε schedule.
5. **Double DQN extension:** the online network selects the next action and the target network evaluates it, reducing overestimation bias. Includes a side-by-side comparison and a recorded gameplay video of the better agent.
6. **Reflection**, references and a statement of work.

## Changes since submission

This version was cleaned up after the module was graded:

- The ε schedule now fits the training budget (10,000 random frames, then decay to 0.1 over 100,000 frames). Previously ε decayed over 1,000,000 frames but training stopped near 125,000, so the agents were still acting almost randomly.
- Only a true game over stops bootstrapping in the Bellman target; time-limit truncation no longer counts as terminal.
- Learning starts only after the replay buffer has been filled with random experience.
- `model.predict()` calls inside the training loops were replaced with direct model calls, which removes a large per-step overhead.
- The gameplay recording now uses the same environment settings as training, picks the better of the two agents and comes after the results.
- Removed duplicated cells and imports, and fixed typos.

Outputs from the original run were cleared, because they came from the old configuration. Run the notebook to regenerate the plots and video.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook dqn_ddqn_space_invaders.ipynb
```

Each agent trains for 250 episodes (about 125,000 frames). This takes hours on a CPU, so a GPU is recommended. Lower `MAX_EPISODES` for a quick test.
