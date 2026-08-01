# Playing pacman using Reinforecement Learning

This code trains a reinforcement learning agent to play PacMan by using only the pixels on the screen.

This repository contains two models:
- A vanilla Deep Q-Network with experience replay
- An enhanced Deep Q-Network with experience replay, Double DQN weights and uses a Dueling architecture

The deep neural net is modeled in tensorflow and we use the Open AI Gym to generate the game simulation. We also use OpenAI baselines - which is a wrapper over tensorflow - for writing the model.

### Requirements:

- Python 3.5
- numpy
- scikit-learn
- tensorflow==1.4.0
- Open AI Gym ([See this link](https://github.com/openai/gym#installation))
- Open AI Gym[atari] ([See this link](https://github.com/openai/gym#atari))
- baselines==0.1.4 ([See this link](https://github.com/openai/baselines). In case of installation issues, read the section below)



### Installation issues:
- Open AI baselines needs mujoco-py installed which in-turn needs Mujoco 1.5 which is a proprietary software. You may be able to install baselines without running into any issues with mujoco-py but if the install of baselines fails dues to mujoco-py, then you need to install Mujoco 1.5 first. The installation instructions are provided on [OpenAI's mujoco-py GitHub repo](https://github.com/openai/mujoco-py). You need to install Mujoco and enter the activation code that can be found from Mujoco's website. This enables a 30 day trial.
- If you are facing problems in the installation of Mujoco 1.5, then mujoco-py==0.5.7 installation can also work but it needs an older version of baselines (0.1.3). This is a breaking change and the code provided in the solution needs some small modifications to make it work.

### Code structure

The code has three directories:
- `python_full_DQN`
- `python_vanilla_DQN`
- `training_logs`

Both directories contain a `pacman_agent.py` file which implements the RL agent, a `save_model` directory which contains a pre-trained model and a `logs` directory which contains execution logs in csv format.

The `training_logs` directory contains some renamed log files for earlier training sessions that have recoreded the step count, episode count and the episode rewards for different values of hyperparameters.

### Training

To train, go into any one of the `python_*_DQN` directories and open the file `pacman_agent.py`. On the top section of the file, you can set any hyperparameters of the model. If you want to continue training from the last saved checkpoint, set the flag `is_load_model=True` otherwise set it to `False`. Then, run:

```
$ python pacman_agent.py
```
This will commence training of the agent. If a saved model checkpoint is loaded, the training will continue from where it left off. Otherwise, the training will begin from scratch. Make sure to change `NUM_STEPS` variable to a large enough number so that the training doesn't stop before you want it to, especially if you're loading an old checkpoint.

(Note: It takes upto 15 hours of training for Full DQN and upto 6 hours of training for vanilla DQN to complete 2,000,000 time steps on a machine with an Nvidia 860m Grapics card, an 8GB RAM and a Core i7 CPU. The training can be continued for far longer than 2 million time steps for better results).

### Watching the agent play

In either of the `python_*_DQN` directories, open the `pacman_agent.py` file and the set the flag `watch_train=True` and `is_load_model=True`. This will load the pretrained model in memory and play the game based on the the Q-values predicted by the model. This will also keep training the model as the more games are played so that the model can improve in the background.

Both `python_*_DQN` directories contain a CSV file, called `play_500_eps.csv` which contains reward data from playing PacMan for 500 episodes. This data is then used to calculate metrics to compare the two models.

### Visualizing metrics

The progress made on training or playing the game can be visualized using the CSV log files. These files store three key data points: the number of time steps, the number of episodes passed and the reward per episode. The easiest visualization is to make a line plot of the number of steps vs the episode reward. This graph is usually very jittery therefore taking a moving average over a fixed number of steps can be helpful in visualizing long term trends.

### Experiment Tracking and Cloud Runs

The modern PyTorch agent (`pacman_full_DQN/pacman_agent_torch.py`) is instrumented with [Weights & Biases](https://wandb.ai) (entity `shehio`, project `pacman-rl`). The legacy TF1.x agents are untouched. What gets logged:

- All hyperparameters (learning rate, target update frequency, replay settings, Double DQN on/off, etc.) as the run config.
- Per episode, at the episode-end hook that already wrote `logs/progress.csv`: `episode_reward`, `episode`, `episode_length`, `episode_reward_avg10`, `epsilon`.
- Per training update, inside the `learn_freq` block (throttled to every 50th update): `loss`, `td_error`, `epsilon`, `beta`.
- End-of-run summary: `final_step`, `final_episode`, `episodes_completed`, `best_episode_reward`, `mean_episode_reward`, `final_episode_reward_avg10`.

New CLI flags on the torch agent: `--lr`, `--max-steps`, `--update-freq`, `--learn-start`, `--replay-size`, `--double-dqn`, `--no-wandb`, `--no-watch`, `--no-load`, `--no-save`, `--wandb-tags`. Note: the README above describes the full model as using Double DQN, but the torch rewrite computes vanilla max targets; `--double-dqn` opts into true Double DQN targets so the two can be A/B tested.

Two flags exist because the defaults are expensive. Each replay entry holds two float64 frame stacks (about 430 KB), so the default capacity of 10000 needs roughly 4 GB of RAM, and `--replay-size` shrinks it for short runs. `--no-save` skips the checkpoint, which otherwise pickles that whole buffer to disk.

Run locally with tracking (install `requirements.txt` first, and `wandb login` once):

```
$ cd pacman_full_DQN
$ python pacman_agent_torch.py --lr 0.00025 --no-watch --no-load
```

Run without wandb (offline or not installed): add `--no-wandb`, or set `WANDB_MODE=disabled`. Either way the CSV logging still happens, so nothing depends on wandb being available. `logs/progress.csv` is a per-run artifact, so it is gitignored going forward and wandb is the durable record.

One caveat when reading short runs: the exploration schedule holds epsilon at 1.0 for the first 10000 steps, so a run shorter than that is pure random play and `episode_reward` will look the same no matter what you changed. Use `loss` and `td_error` to sanity check a short run, and `episode_reward_avg10` only once training has actually started acting greedily.

**Sweeps.** Two sweep configs live under `sweeps/`. `dqn_grid.yaml` is the full grid over learning rate (5e-3, 1e-4, 2.5e-4), target update frequency (100, 1000, 5000, 10000), and Double DQN on/off, which is 24 runs. `double_dqn_ab.yaml` is the focused two-run A/B that isolates the Double DQN question at a fixed learning rate. Launch either the same way:

```
$ wandb sweep sweeps/dqn_grid.yaml
$ wandb agent shehio/pacman-rl/<sweep-id>
```

**Modal.** `modal_app.py` runs training on an A10G GPU in the cloud. One-time setup: `pip install modal`, `modal token new`, then create a Modal secret named `wandb` holding your `WANDB_API_KEY`. Then:

```
$ modal run modal_app.py --lr 0.00025 --double-dqn --max-steps 2000000
```
























