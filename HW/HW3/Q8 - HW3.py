import random

gamma = 0.8
episodes = 50
actions = ['P', 'R', 'S', 'any']
rewards = [2, 0, -1]
states = ['RU_8p', 'TU_10p', 'RU_10p', 'RD_10p', 'RU_8a', 'RD_8a', 'TU_10a', 'RU_10a', 'RD_10a', 'TD_10a', 'Terminal']

return_episodes = []

for i in range(episodes):
    trajectory = ['RU_8p']
    action_index = random.randint(0, len(actions) - 2)
    trajectory.extend([actions[action_index], rewards[action_index]])

    # Determine next state based on the action
    if trajectory[1] == 'P':
        trajectory.append(states[1])
    elif trajectory[1] == 'R':
        trajectory.append(states[2])
    elif trajectory[1] == 'S':
        trajectory.append(states[3])

    current_state = trajectory[3]

    # Update trajectory based on current state
    if current_state == 'TU_10p':
        action_index = random.randint(0, len(actions) - 3)
        state = 'RU_10a' if actions[action_index] == 'P' else 'RU_8a'
    elif current_state == 'RU_10p':
        action_index = random.randint(0, len(actions) - 2)
        state = ('RU_8a' if actions[action_index] == 'R' else
                 'RD_8a' if actions[action_index] == 'S' else
                 random.choice(['RU_8a', 'RU_10a']))
    elif current_state == 'RD_10p':
        action_index = random.randint(0, len(actions) - 3)
        state = 'RD_8a' if actions[action_index] == 'R' else random.choice(['RD_8a', 'RD_10a'])

    reward = rewards[action_index]
    action = actions[action_index]
    trajectory.extend([action, reward, state])

    # Further state transitions
    if trajectory[6] == 'RU_8a':
        action_index = random.randint(0, len(actions) - 2)
        state = ('RU_10a' if actions[action_index] == 'R' else
                 'RD_10a' if actions[action_index] == 'S' else
                 'TU_10a')
    elif trajectory[6] == 'RD_8a':
        action_index = random.randint(0, len(actions) - 3)
        state = 'RD_10a' if actions[action_index] == 'R' else 'TD_10a'
    elif trajectory[6] in ['RU_10a', 'RD_10a']:
        state = 'Terminal'
        action_index = 3
        reward = 0 if trajectory[6] == 'RU_10a' else 4

    trajectory.extend([actions[action_index], reward, state])

    if trajectory[9] == 'Terminal':
        print("Trajectory:")
        print(trajectory)
        episode_return = trajectory[2] + gamma * trajectory[5] + gamma ** 2 * trajectory[8]
        print(f"Return of episode {i + 1}:")
        return_episodes.append(episode_return)
        print(episode_return)
        continue

    # Additional transitions if not terminal
    if trajectory[9] in ['TU_10a', 'RU_10a', 'RD_10a', 'TD_10a']:
        state = 'Terminal'
        action_index = 3
        reward = {'TU_10a': -1, 'RU_10a': 0, 'RD_10a': 4, 'TD_10a': 3}[trajectory[9]]

    trajectory.extend([actions[action_index], reward, state])
    print("Trajectory:")
    print(trajectory)
    episode_return = (trajectory[2] + gamma * trajectory[5] +
                      gamma ** 2 * trajectory[8] + gamma ** 3 * trajectory[11])
    print(f"Return of episode {i + 1}:")
    return_episodes.append(episode_return)
    print(episode_return)

# Print the average return across all episodes
mean_return = sum(return_episodes) / len(return_episodes)
print("Mean return of all episodes:")
print(mean_return)
