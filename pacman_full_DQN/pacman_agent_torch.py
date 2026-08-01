import gymnasium as gym
import ale_py  # This registers the ALE environments
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from skimage import color
from skimage.transform import resize
import csv
import os
import random
import argparse

try:
    import wandb
except ImportError:  # wandb is optional; run with --no-wandb when it is not installed
    wandb = None

# Environment and preprocessing setup (copied from original)
env = gym.make("ALE/MsPacman-v5")
input_height, input_width = (86, 80)
batch_size = 32
update_freq = 10000
learn_freq = 4
save_freq = 500000
action_space_size = env.action_space.n
NUM_STEPS = 20000000
replay_memory_size = 10000
replay_alpha = 0.6
replay_beta = 0.4
replay_epsilon = 1e-6
is_load_model = True
watch_flag = True
fps = 30 #frames shown per second when watch_flag == True

# Logging setup
def log_to_csv(log_path, log_dict):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(log_path)
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_dict)

log_path = os.path.join("logs", "progress.csv")

# Preprocessing function
def preprocess_frame(frame):
    im = resize(color.rgb2gray(frame)[:176, :], (input_height, input_width), mode='constant')
    return im

def build_state(state_deque):
    """Stack frames together to construct state"""
    return np.array(list(state_deque)) 

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingDQN, self).__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Compute conv output size
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        # State value stream
        self.fc1_state = nn.Linear(linear_input_size, 512)
        self.bn_state = nn.BatchNorm1d(512)
        self.fc2_state = nn.Linear(512, 1)
        
        # Action advantage stream
        self.fc1_adv = nn.Linear(linear_input_size, 512)
        self.bn_adv = nn.BatchNorm1d(512)
        self.fc2_adv = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x
        x = x / 255.0  # normalize if input is uint8
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        # State value
        state = self.fc1_state(x)
        state = F.relu(state)
        state = self.bn_state(state)
        state = self.fc2_state(state)
        # Action advantage
        adv = self.fc1_adv(x)
        adv = F.relu(adv)
        adv = self.bn_adv(adv)
        adv = self.fc2_adv(adv)
        adv_mean = adv.mean(1, keepdim=True)
        adv = adv - adv_mean
        # Combine
        qvals = state + adv
        return qvals 

class PrioritizedReplayBuffer:
    def __init__(self, size, alpha):
        self._maxsize = size
        self._alpha = alpha
        self._next_idx = 0
        self._storage = []
        self._priorities = np.zeros((size,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        max_prio = self._priorities.max() if self._storage else 1.0
        if len(self._storage) < self._maxsize:
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._priorities[self._next_idx] = max_prio
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size, beta=0.4):
        if len(self._storage) == self._maxsize:
            prios = self._priorities
        else:
            prios = self._priorities[:self._next_idx]
        probs = prios ** self._alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self._storage), batch_size, p=probs)
        samples = [self._storage[idx] for idx in indices]
        total = len(self._storage)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        batch = list(zip(*samples))
        return (
            np.array(batch[0]),  # states
            np.array(batch[1]),  # actions
            np.array(batch[2]),  # rewards
            np.array(batch[3]),  # next_states
            np.array(batch[4]),  # dones
            np.array(weights, dtype=np.float32),
            np.array(indices)
        )

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self._priorities[idx] = prio

    def __len__(self):
        return len(self._storage)

# Schedules
class LinearSchedule:
    def __init__(self, schedule_timesteps, initial_p, final_p):
        self.schedule_timesteps = schedule_timesteps
        self.initial_p = initial_p
        self.final_p = final_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class PiecewiseSchedule:
    def __init__(self, endpoints, outside_value):
        self.endpoints = endpoints
        self.outside_value = outside_value

    def value(self, t):
        for (l_t, l), (r_t, r) in zip(self.endpoints[:-1], self.endpoints[1:]):
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return l + alpha * (r - l)
        return self.outside_value 

def save_model(model, replay_memory, dq, step, path_prefix="saved_model/"):
    os.makedirs(path_prefix, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'replay_memory': replay_memory,
        'dq': dq,
        'step': step
    }, os.path.join(path_prefix, "model.pt"))

def load_model(model, path_prefix="saved_model/"):
    checkpoint = torch.load(os.path.join(path_prefix, "model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['replay_memory'], checkpoint['dq'], checkpoint['step']

def parse_args():
    parser = argparse.ArgumentParser(description="Dueling DQN on ALE/MsPacman-v5 (PyTorch)")
    parser.add_argument("--lr", type=float, default=0.005, help="Adam learning rate")
    parser.add_argument("--max-steps", type=int, default=NUM_STEPS, help="total environment steps to train for")
    parser.add_argument("--update-freq", type=int, default=update_freq, help="target network sync frequency (steps)")
    parser.add_argument("--learn-start", type=int, default=10000, help="steps before gradient updates begin")
    parser.add_argument("--replay-size", type=int, default=replay_memory_size,
                        help="prioritized replay capacity (transitions); each one holds two float64 frame "
                             "stacks, roughly 430 KB, so the default 10000 needs about 4 GB of RAM")
    parser.add_argument("--double-dqn", action="store_true",
                        help="use Double DQN targets (default: vanilla max targets, the current behavior)")
    parser.add_argument("--no-wandb", action="store_true", help="disable Weights & Biases logging")
    parser.add_argument("--no-watch", action="store_true", help="disable rendering even if watch_flag is set")
    parser.add_argument("--no-load", action="store_true", help="start fresh even if a saved model exists")
    parser.add_argument("--no-save", action="store_true",
                        help="skip checkpointing (useful for sweeps and smoke runs, the replay buffer is large)")
    parser.add_argument("--wandb-tags", type=str, default="", help="comma-separated wandb tags")
    return parser.parse_args()

def main(args=None):
    if args is None:
        args = parse_args()
    watch = watch_flag and not args.no_watch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_wandb = wandb is not None and not args.no_wandb
    run = None
    if use_wandb:
        tags = [t for t in args.wandb_tags.split(",") if t]
        run = wandb.init(
            entity="shehio",
            project="pacman-rl",
            config={
                "env": "ALE/MsPacman-v5",
                "lr": args.lr,
                "max_steps": args.max_steps,
                "batch_size": batch_size,
                "update_freq": args.update_freq,
                "learn_freq": learn_freq,
                "learn_start": args.learn_start,
                "save_freq": save_freq,
                "replay_memory_size": args.replay_size,
                "replay_alpha": replay_alpha,
                "replay_beta": replay_beta,
                "replay_epsilon": replay_epsilon,
                "double_dqn": args.double_dqn,
                "gamma": 0.99,
                "grad_clip": 10,
                "input_height": input_height,
                "input_width": input_width,
                "frame_stack": 4,
                "device": str(device),
            },
            tags=tags or None,
        )
    model = DuelingDQN((4, input_height, input_width), action_space_size).to(device)
    target_model = DuelingDQN((4, input_height, input_width), action_space_size).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    epsilon = PiecewiseSchedule([(0, 1.0),
                                 (10000, 1.0),
                                 (20000, 0.4),
                                 (50000, 0.2),
                                 (100000, 0.1),
                                 (500000, 0.05)], outside_value=0.01)
    replay_memory = PrioritizedReplayBuffer(args.replay_size, replay_alpha)
    beta = LinearSchedule(int(args.max_steps/4), initial_p=replay_beta, final_p=1.0)
    dq = []
    start_step = 1
    episode = 1
    if is_load_model and not args.no_load and os.path.exists("saved_model/model.pt"):
        replay_memory, dq, start_step = load_model(model)
    episode_rewards = []
    episode_start_step = start_step
    obs, _ = env.reset()
    state = preprocess_frame(obs)
    state_deque = deque([state]*4, maxlen=4)
    state_full = build_state(state_deque)
    for step in itertools.count(start=start_step):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon.value(step):
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                model.eval()
                state_tensor = torch.tensor(state_full, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = model(state_tensor)
                action = q_values.argmax(1).item()
                model.train()
        obs_tplus1, reward, terminated, truncated, _ = env.step(action)
        is_finished = terminated or truncated
        state_tplus1 = preprocess_frame(obs_tplus1)
        state_deque.append(state_tplus1)
        state_tplus1_full = build_state(state_deque)
        dq.append(reward)
        if watch:
            env.render()
            time.sleep(1.0/fps)
        replay_memory.add(state_full, action, reward, state_tplus1_full, float(is_finished))
        state_full = state_tplus1_full
        if is_finished:
            ep_reward = sum(dq)
            ep_length = step - episode_start_step + 1
            episode_rewards.append(ep_reward)
            log_to_csv(log_path, {"Steps": step, "Episode reward": ep_reward, "Episode number": episode})
            if use_wandb:
                wandb.log({"episode_reward": ep_reward,
                           "episode": episode,
                           "episode_length": ep_length,
                           "episode_reward_avg10": sum(episode_rewards[-10:]) / len(episode_rewards[-10:]),
                           "epsilon": epsilon.value(step)}, step=step)
            print(f"Step {step}. Finished episode {episode} with reward {ep_reward}")
            dq = []
            episode_start_step = step + 1
            obs, _ = env.reset()
            state = preprocess_frame(obs)
            state_deque = deque([state]*4, maxlen=4)
            state_full = build_state(state_deque)
            episode += 1
            for _ in range(30):
                env.step(0)
                if watch:
                    env.render()
        if step > args.learn_start and step % learn_freq == 0:
            batch = replay_memory.sample(batch_size, beta=beta.value(step))
            states = torch.tensor(batch[0], dtype=torch.float32, device=device)
            actions = torch.tensor(batch[1], dtype=torch.int64, device=device)
            rewards = torch.tensor(batch[2], dtype=torch.float32, device=device)
            next_states = torch.tensor(batch[3], dtype=torch.float32, device=device)
            dones = torch.tensor(batch[4], dtype=torch.float32, device=device)
            weights = torch.tensor(batch[5], dtype=torch.float32, device=device)
            indices = batch[6]
            # Compute Q targets
            q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                if args.double_dqn:
                    # Double DQN: online net picks the action, target net evaluates it
                    next_actions = model(next_states).argmax(1, keepdim=True)
                    next_q_values = target_model(next_states).gather(1, next_actions).squeeze(1)
                else:
                    next_q_values = target_model(next_states).max(1)[0]
                targets = rewards + (1 - dones) * 0.99 * next_q_values
            td_errors = q_values - targets
            loss = (weights * td_errors.pow(2)).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            priorities = (td_errors.abs().cpu().detach().numpy() + replay_epsilon)
            replay_memory.update_priorities(indices, priorities)
            if use_wandb and step % (learn_freq * 50) == 0:
                wandb.log({"loss": loss.item(),
                           "td_error": td_errors.abs().mean().item(),
                           "epsilon": epsilon.value(step),
                           "beta": beta.value(step)}, step=step)
        if step % args.update_freq == 0:
            target_model.load_state_dict(model.state_dict())
        if step % save_freq == 0 and not args.no_save:
            print(f"State save {step}")
            save_model(model, replay_memory, dq, step)
        if step > args.max_steps:
            if args.no_save:
                print("Finished training. Skipping checkpoint (--no-save).")
            else:
                print("Finished training. Saving model to ./saved_model/model.pt")
                save_model(model, replay_memory, dq, step)
            if use_wandb:
                run.summary["final_step"] = step
                run.summary["final_episode"] = episode
                run.summary["episodes_completed"] = len(episode_rewards)
                if episode_rewards:
                    run.summary["best_episode_reward"] = max(episode_rewards)
                    run.summary["mean_episode_reward"] = sum(episode_rewards) / len(episode_rewards)
                    run.summary["final_episode_reward_avg10"] = (
                        sum(episode_rewards[-10:]) / len(episode_rewards[-10:]))
                run.finish()
            break

if __name__ == "__main__":
    main() 