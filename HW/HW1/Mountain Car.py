import gymnasium as gym
import matplotlib.pyplot as plt


env = gym.make("MountainCar-v0", render_mode="rgb_array")


observation, info = env.reset()


positions = []


for _ in range(10000):
    action = env.action_space.sample() 
    observation, reward, done, truncated, info = env.step(action)  
    position = observation[0] 
    positions.append(position) 
    
    if done or truncated:
        observation, info = env.reset()  

env.close()


plt.plot(positions)
plt.title('Car Position Over 10000 Random Actions')
plt.xlabel('Step')
plt.ylabel('Position')
plt.show()
