import gymnasium as gym
import matplotlib.pyplot as plt

# ایجاد محیط Car Mountain
env = gym.make("MountainCar-v0", render_mode="rgb_array")

# راه‌اندازی مجدد محیط
observation, info = env.reset()

# ذخیره داده‌ها برای موقعیت‌های خودرو
positions = []

# اجرای 10000 عمل تصادفی
for _ in range(10000):
    action = env.action_space.sample()  # انتخاب عمل تصادفی
    observation, reward, done, truncated, info = env.step(action)  # انجام عمل
    position = observation[0]  # موقعیت خودرو
    positions.append(position)  # ذخیره موقعیت
    
    if done or truncated:
        observation, info = env.reset()  # راه‌اندازی مجدد محیط در صورت اتمام اپیزود

env.close()

# نمایش نمودار تغییرات موقعیت خودرو
plt.plot(positions)
plt.title('Car Position Over 10000 Random Actions')
plt.xlabel('Step')
plt.ylabel('Position')
plt.show()
