from os import name
import gymnasium as gym

ENV_NAME = 'CartPole-v1'


def make_env():
    env = gym.make(ENV_NAME, render_mode='human')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    return env, obs_dim, act_dim

if __name__ == "__main__":
    env, obs_dim, act_dim = make_env()
    
    # Reset the environment
    obs, info = env.reset()
    
    # Render the environment (opens a window)
    env.render()
    
    # Take a random action to see it move
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Keep rendering for a few steps
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()