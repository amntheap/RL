import gymnasium as gym
import numpy as np
from collections import defaultdict

#   P1 -----------------------------------------------------------------------------------------

# Initialize environment
env = gym.make("FrozenLake-v1", is_slippery=False).unwrapped
n_states = env.observation_space.n
n_actions = env.action_space.n
gamma = 0.99  # Discount factor
theta = 1e-8  # Convergence threshold for policy evaluation

# Helper function to display the policy as a 4x4 grid with directional arrows
def display_policy(policy):
    action_symbols = ["←", "↓", "→", "↑"]
    print("\nOptimal Policy (4x4 Grid):")
    for i in range(0, n_states, 4):
        print([action_symbols[policy[s]] for s in range(i, i + 4)])

# Helper function to display the value function as a 4x4 grid
def display_value_function(V):
    print("\nOptimal Value Function (4x4 Grid):")
    for i in range(0, len(V), 4):
        print([f"{V[j]:.2f}" for j in range(i, i + 4)])

# Policy Iteration Algorithm
def policy_iteration(env, gamma=0.99, theta=1e-8):
    policy = np.random.randint(0, n_actions, n_states)  # Initialize a random policy
    V = np.zeros(n_states)  # Value function initialized to zero
    
    while True:
        # Policy Evaluation Step
        while True:
            delta = 0
            for s in range(n_states):
                if s == n_states - 1:  # Skip terminal states
                    continue
                
                v = sum(prob * (reward + gamma * V[next_state]) 
                        for prob, next_state, reward, done in env.P[s][policy[s]])
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            
            if delta < theta:
                break
        
        # Policy Improvement Step
        policy_stable = True
        for s in range(n_states):
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                action_values[a] = sum(prob * (reward + gamma * V[next_state]) 
                                       for prob, next_state, reward, done in env.P[s][a])
            best_action = np.argmax(action_values)
            
            if best_action != policy[s]:
                policy_stable = False
            policy[s] = best_action
        
        if policy_stable:
            break
    
    return policy, V

# Find the optimal policy and value function using Policy Iteration
optimal_policy_pi, optimal_values_pi = policy_iteration(env, gamma, theta)

# Display the final optimal policy and value function
print("\nPolicy Iteration Results:")
display_policy(optimal_policy_pi)
display_value_function(optimal_values_pi)

# Close the environment
env.close()


#   V1 --------------------------------------------------------------------------------------------------------------------

# Initialize environment
env = gym.make("FrozenLake-v1", is_slippery=False).unwrapped
n_states = env.observation_space.n
n_actions = env.action_space.n
gamma = 0.99  # Discount factor
theta = 1e-8  # Convergence threshold for value function updates

# Helper function to display the policy as a 4x4 grid with directional arrows
def display_policy(policy):
    action_symbols = ["←", "↓", "→", "↑"]
    print("\nOptimal Policy (4x4 Grid):")
    for i in range(0, n_states, 4):
        print([action_symbols[policy[s]] for s in range(i, i + 4)])

# Helper function to display the value function as a 4x4 grid
def display_value_function(V):
    print("\nOptimal Value Function (4x4 Grid):")
    for i in range(0, len(V), 4):
        print([f"{V[j]:.2f}" for j in range(i, i + 4)])

# Value Iteration Algorithm
def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(n_states)  # Initialize value function to zero
    
    while True:
        delta = 0
        for s in range(n_states):
            if s == n_states - 1:  # Skip terminal states
                continue
            
            action_values = []
            for a in range(n_actions):
                value = sum(prob * (reward + gamma * V[next_state]) 
                            for prob, next_state, reward, done in env.P[s][a])
                action_values.append(value)
            
            max_value = max(action_values)
            delta = max(delta, abs(max_value - V[s]))
            V[s] = max_value
        
        if delta < theta:
            break
    
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            action_values[a] = sum(prob * (reward + gamma * V[next_state]) 
                                   for prob, next_state, reward, done in env.P[s][a])
        policy[s] = np.argmax(action_values)
    
    return policy, V

# Find the optimal policy and value function using Value Iteration
optimal_policy_vi, optimal_values_vi = value_iteration(env, gamma, theta)

# Display the final optimal policy and value function
print("Value Iteration Results:")
display_policy(optimal_policy_vi)
display_value_function(optimal_values_vi)

# Close the environment
env.close()


#   on-policy -------------------------------------------------------------------------------------------------------------

# Initialize environment
env = gym.make("FrozenLake-v1", is_slippery=False).unwrapped
n_states = env.observation_space.n
n_actions = env.action_space.n
gamma = 0.99  # Discount factor
max_steps = 200  # Maximum steps per episode

# Helper function to display the policy as a 4x4 grid with directional arrows
def display_policy(policy):
    action_symbols = ["←", "↓", "→", "↑"]
    print("\nCurrent Policy:")
    for i in range(0, n_states, 4):
        print([action_symbols[policy[s]] for s in range(i, i + 4)])

# On-policy Monte Carlo Control with Epsilon-Greedy Exploration (Every-Visit)
def monte_carlo_on_policy(n_episodes=50000, initial_epsilon=0.5, min_epsilon=0.1, epsilon_decay_interval=10000, epsilon_decay_amount=0.1, print_interval=5000):
    # Initialize Q-table and policy randomly
    Q = defaultdict(lambda: np.zeros(n_actions))
    policy = np.random.randint(0, n_actions, n_states)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    
    # Epsilon decay schedule
    epsilon = initial_epsilon

    # Epsilon-greedy action selection function
    def epsilon_greedy_policy(state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(n_actions)  # Explore
        return np.argmax(Q[state])  # Exploit the best known action

    # Variables to track performance over time
    total_rewards = 0
    successful_episodes = 0

    # Run episodes
    for episode in range(1, n_episodes + 1):
        episode_data = []
        state, _ = env.reset()
        episode_reward = 0

        # Generate an episode following epsilon-greedy policy
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = epsilon_greedy_policy(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            episode_data.append((state, action, reward))  # Default reward structure
            episode_reward += reward
            state = next_state
            steps += 1

        # Track total rewards and successes
        total_rewards += episode_reward
        if episode_reward > 0:  # Success if the agent reached the goal
            successful_episodes += 1

        # Calculate returns and update Q-values (Every-Visit MC)
        G = 0
        for t in reversed(range(len(episode_data))):
            state, action, reward = episode_data[t]
            G = gamma * G + reward  # Discounted return

            # Every-Visit MC: Update Q-value for every occurrence of (state, action)
            returns_sum[(state, action)] += G
            returns_count[(state, action)] += 1
            Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

            # Improve policy by choosing the action with the highest Q-value
            policy[state] = np.argmax(Q[state])

        # Update epsilon in a fixed interval
        if episode % epsilon_decay_interval == 0 and epsilon > min_epsilon:
            epsilon = max(epsilon - epsilon_decay_amount, min_epsilon)

        # Print progress every `print_interval` episodes
        if episode % print_interval == 0:
            success_rate = (successful_episodes / print_interval) * 100
            avg_reward = total_rewards / print_interval
            print(f"\nEpisode {episode}/{n_episodes}")
            print(f"Success Rate: {success_rate:.2f}%")
            print(f"Average Reward per Episode: {avg_reward:.4f}")
            display_policy(policy)
            # Reset tracking for next interval
            total_rewards = 0
            successful_episodes = 0

    return policy

# Train the on-policy Monte Carlo Control algorithm with monitoring
optimal_policy = monte_carlo_on_policy(
    n_episodes=50000, 
    initial_epsilon=0.5, 
    min_epsilon=0.1, 
    epsilon_decay_interval=10000, 
    epsilon_decay_amount=0.1, 
    print_interval=5000
)

# Display the final optimal policy
print("\nFinal Optimal Policy:")
display_policy(optimal_policy)

# Close the environment
env.close()



#   off-policy -------------------------------------------------------------------------------------------------------------

# Initialize environment
env = gym.make("FrozenLake-v1", is_slippery=False).unwrapped
n_states = env.observation_space.n
n_actions = env.action_space.n
gamma = 0.99  # Discount factor
max_steps = 200  # Maximum steps per episode

# Helper function to display the policy as a 4x4 grid with directional arrows
def display_policy(policy):
    action_symbols = ["←", "↓", "→", "↑"]
    print("\nCurrent Policy:")
    for i in range(0, n_states, 4):
        print([action_symbols[policy[s]] for s in range(i, i + 4)])

# Off-policy Monte Carlo Control with Weighted Importance Sampling
def monte_carlo_off_policy(n_episodes=50000, epsilon=0.5, min_epsilon=0.1, epsilon_decay_interval=10000, epsilon_decay_amount=0.1, print_interval=5000):
    # Initialize Q-table and policy randomly
    Q = defaultdict(lambda: np.zeros(n_actions))
    target_policy = np.random.randint(0, n_actions, n_states)  # Initialize target policy
    C = defaultdict(lambda: np.zeros(n_actions))  # Cumulative sum of weights for each state-action pair

    # Epsilon-greedy action selection function for behavior policy
    def epsilon_greedy_behavior_policy(state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(n_actions)  # Explore
        return np.argmax(Q[state])  # Exploit the best-known action

    # Variables to track performance over time
    total_rewards = 0
    successful_episodes = 0

    # Run episodes
    for episode in range(1, n_episodes + 1):
        episode_data = []
        state, _ = env.reset()
        episode_reward = 0
        epsilon_current = max(epsilon - epsilon_decay_amount * (episode // epsilon_decay_interval), min_epsilon)

        # Generate an episode following epsilon-greedy behavior policy
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = epsilon_greedy_behavior_policy(state, epsilon_current)
            next_state, reward, done, _, _ = env.step(action)
            episode_data.append((state, action, reward))
            episode_reward += reward
            state = next_state
            steps += 1

        # Track total rewards and successes
        total_rewards += episode_reward
        if episode_reward > 0:  # Success if the agent reached the goal
            successful_episodes += 1

        # Calculate returns and update Q-values using Weighted Importance Sampling
        G = 0
        W = 1  # Initial importance sampling weight

        for state, action, reward in reversed(episode_data):
            G = gamma * G + reward  # Accumulate return

            # Update cumulative weight and Q-value if weight is non-zero
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

            # Improve the target policy by choosing the action with the highest Q-value
            target_policy[state] = np.argmax(Q[state])

            # Update the importance sampling weight
            if action != target_policy[state]:  # If the action deviates from the target policy, stop weighting
                break
            W *= 1 / (epsilon_current / n_actions + (1 - epsilon_current) if action == target_policy[state] else epsilon_current / n_actions)

        # Print progress every `print_interval` episodes
        if episode % print_interval == 0:
            success_rate = (successful_episodes / print_interval) * 100
            avg_reward = total_rewards / print_interval
            print(f"\nEpisode {episode}/{n_episodes}")
            print(f"Success Rate: {success_rate:.2f}%")
            print(f"Average Reward per Episode: {avg_reward:.4f}")
            display_policy(target_policy)
            # Reset tracking for next interval
            total_rewards = 0
            successful_episodes = 0

    return target_policy

# Train the off-policy Monte Carlo Control algorithm with monitoring
optimal_policy = monte_carlo_off_policy(
    n_episodes=50000,
    epsilon=0.5,
    min_epsilon=0.1,
    epsilon_decay_interval=10000,
    epsilon_decay_amount=0.1,
    print_interval=5000
)

# Display the final optimal policy
print("\nFinal Optimal Policy:")
display_policy(optimal_policy)

# Close the environment
env.close()
