import gymnasium as gym
import numpy as np
import time

env = gym.make("FrozenLake-v1", is_slippery=False)
q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.8
gamma = 0.95
eps = 1.0

for episode in range(3000):
    state, _ = env.reset()
    done = False
    total_hadiah = 0

    while not done:

        if np.random.rand() < eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state])

        next_state, hadiah, terminated, truncated, _ = env.step(action)

        q[state, action] += alpha * (hadiah + gamma * np.max(q[next_state]) - q[state, action])

        state = next_state
        total_hadiah += hadiah

        done = terminated or truncated

    eps *= 0.99

    print(f"Episode {episode + 1}: Hadiah = {total_hadiah}")

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
state, _ = env.reset()
done = False

print("\nMenjalankan demo dengan GUI...")
while not done:
    env.render()
    time.sleep(0.4)
    action = np.argmax(q[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

print(f"Demo selesai! Hadiah: {hadiah}")
env.close()