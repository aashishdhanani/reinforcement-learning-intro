import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import wandb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import argparse
import os
from datetime import datetime


ENV_NAME = 'CartPole-v1'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.act_dim_ = act_dim
        self.obs_dim_ = obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        
        action_onehot = torch.nn.functional.one_hot(action, num_classes=self.act_dim_).float()
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)
    
    def sample_a(self, state, epsilon=0.1):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.act_dim_, (1,), device=state.device).item()
        else:
            action = self.get_max_a(state)
            return action if isinstance(action, int) else action.item()
    
    def get_max_a(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        batch_size = state.size(0)
        actions = torch.arange(self.act_dim_, device=state.device)
        actions = actions.unsqueeze(0).repeat(batch_size, 1)  # [B, A]
        states = state.unsqueeze(1).repeat(1, self.act_dim_, 1)  # [B, A, S]
        
        q_values = self.forward(states.view(-1, state.size(-1)), actions.view(-1))
        q_values = q_values.view(batch_size, self.act_dim_)
        best_action = torch.argmax(q_values, dim=1)
        return best_action.item() if batch_size == 1 else best_action
    
    @property
    def act_dim(self):
        return self.act_dim_
    
    @property
    def obs_dim(self):
        return self.obs_dim_


def make_env():
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    return env, obs_dim, act_dim


def train(run_id, config):
    # Initialize wandb
    run_name = f"run_{run_id}"
    wandb.init(
        project=config.project_name,
        name=run_name,
        group=config.group_name,
        config={
            "env_name": ENV_NAME,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "episodes": config.episodes,
            "epsilon": config.epsilon,
            "run_id": run_id
        }
    )
    
    env, obs_dim, act_dim = make_env()
    q_net = QNetwork(obs_dim, act_dim).to(DEVICE)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.learning_rate)
    gamma = config.gamma
    epsilon = config.epsilon
    
    all_rewards = []
    
    for episode in range(config.episodes):
        state = env.reset()
        if isinstance(state, tuple):  # gymnasium compatibility
            state = state[0]
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        
        done = False
        ep_reward = 0
        step_count = 0
        total_loss = 0
        
        while not done:
            action = q_net.sample_a(state, epsilon)
            next_state, reward, done, info = env.step(action)
            if isinstance(info, dict) and 'TimeLimit.truncated' in info:
                truncated = info.get('TimeLimit.truncated', False)
                done = done and not truncated
            
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            next_state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)
            
            q_val = q_net(state, torch.tensor(action, device=DEVICE))
            with torch.no_grad():
                target = reward + gamma * q_net(next_state, torch.tensor(q_net.get_max_a(next_state), device=DEVICE)) * (1. - float(done))
            
            loss = (q_val - target).pow(2).mean()
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            ep_reward += reward
            step_count += 1
        
        # Calculate metrics
        all_rewards.append(ep_reward)
        avg_reward_10 = sum(all_rewards[-10:]) / min(len(all_rewards), 10)
        avg_loss = total_loss / step_count if step_count > 0 else 0
        
        # Log to wandb
        wandb.log({
            "episode": episode,
            "reward": ep_reward,
            "avg_reward_10": avg_reward_10,
            "loss": avg_loss,
            "steps": step_count
        })
        
        if episode % 10 == 0 or episode == config.episodes - 1:
            print(f"Run {run_id} | Episode {episode} | Reward: {ep_reward:.2f} | Avg(10): {avg_reward_10:.2f}", flush=True)
    
    # Save final model if needed
    if config.save_model:
        model_path = os.path.join(config.output_dir, f"q_net_run_{run_id}.pt")
        torch.save(q_net.state_dict(), model_path)
    
    wandb.finish()
    return all_rewards


def smooth_rewards(rewards, sigma=2):
    """Apply Gaussian smoothing to rewards."""
    return gaussian_filter1d(rewards, sigma=sigma)


def create_summary_plot(all_run_rewards, config):
    """Create and save a summary plot showing average and individual runs."""
    plt.figure(figsize=(12, 8))
    
    # Plot individual runs with transparency
    for i, rewards in enumerate(all_run_rewards):
        episodes = range(1, len(rewards) + 1)
        smoothed = smooth_rewards(rewards)
        plt.plot(episodes, smoothed, alpha=0.3, label=f"Run {i+1}" if i < 5 else "")
    
    # Calculate and plot average
    min_length = min(len(r) for r in all_run_rewards)
    truncated_rewards = [r[:min_length] for r in all_run_rewards]
    avg_rewards = np.mean(truncated_rewards, axis=0)
    smoothed_avg = smooth_rewards(avg_rewards)
    
    plt.plot(range(1, min_length + 1), smoothed_avg, 'k-', linewidth=2, label="Average")
    
    # Add confidence intervals (std dev)
    std_rewards = np.std(truncated_rewards, axis=0)
    plt.fill_between(
        range(1, min_length + 1),
        smoothed_avg - std_rewards,
        smoothed_avg + std_rewards,
        color='k', alpha=0.2, label="±1 Std Dev"
    )
    
    plt.title(f"{ENV_NAME} - DQN Average Performance over {len(all_run_rewards)} Runs")
    plt.xlabel("Episodes")
    plt.ylabel("Smoothed Reward")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(config.output_dir, f"dqn_summary_plot_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Summary plot saved to {filename}")
    
    # Log to wandb as a summary artifact
    summary_run = wandb.init(
        project=config.project_name,
        name="dqn_summary",
        group=config.group_name,
        job_type="analysis"
    )
    
    wandb.log({"summary_plot": wandb.Image(plt)})
    summary_data = {
        "mean_final_reward": float(avg_rewards[-1]),
        "std_final_reward": float(std_rewards[-1]),
        "max_mean_reward": float(np.max(avg_rewards)),
        "episode_of_max": int(np.argmax(avg_rewards) + 1),
        "num_runs": len(all_run_rewards)
    }
    wandb.log(summary_data)
    
    # Create a summary table of statistics
    data = [[i+1, r[-1], np.max(r), np.argmax(r)+1] for i, r in enumerate(all_run_rewards)]
    table = wandb.Table(columns=["Run", "Final Reward", "Max Reward", "Max Episode"], data=data)
    wandb.log({"run_stats": table})
    
    wandb.finish()
    return filename


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent with multiple runs')
    parser.add_argument('--runs', type=int, default=50, help='Number of training runs')
    parser.add_argument('--episodes', type=int, default=300, help='Number of episodes per run')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration rate')
    parser.add_argument('--project_name', type=str, default='dqn_cartpole', help='WandB project name')
    parser.add_argument('--group_name', type=str, default=None, help='WandB group name')
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoints')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs')
    
    config = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    # Set default group name if not specified
    if config.group_name is None:
        config.group_name = f"{ENV_NAME}_DQN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Starting {config.runs} training runs with {config.episodes} episodes each")
    print(f"WandB Project: {config.project_name}, Group: {config.group_name}")
    
    all_run_rewards = []
    
    for run_id in range(1, config.runs + 1):
        print(f"\n=== Starting Run {run_id}/{config.runs} ===")
        run_rewards = train(run_id, config)
        all_run_rewards.append(run_rewards)
    
    # Create summary visualization
    summary_plot = create_summary_plot(all_run_rewards, config)
    print(f"\nTraining complete! Summary visualization saved to {summary_plot}")


if __name__ == "__main__":
    main()