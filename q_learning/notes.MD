initial run error:

- run_rewards = train(run_id, config)
next_state, reward, done, info = env.step(action)

too many values to unpack

what prints when i do print(env):
- <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>

what does env.step actually do?
- run one timestep of env using given action

what does env.step return?
- observation
- reward
- terminated
- truncated
- info
- done

i think i just had to add truncated since that is part of the return tuple

RUNNING NOW, what are the core RL concepts happening while training:
1. Q value learning update --> q_val is the current Q value estimate and target is what it should be
2. The network learns by minimizing this difference

Each step: observe state → take action → get reward → update Q-values

Each run is 300 episodes. Its 50 different agents, each trainied independetly in the same Cartpole env
what is different during each run?
- neural network is random, optimizer state randomized, and random seed/exploration path


main training loop is in here

 for run_id in range(1, config.runs + 1):
     print(f"\n=== Starting Run {run_id}/{config.runs} ===")
     run_rewards = train(run_id, config)
     all_run_rewards.append(run_rewards)