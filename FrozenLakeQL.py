import gym
import numpy as np
from tqdm import tqdm

env = gym.make('FrozenLake-v0')
gamma = 0.99
epsilon = 1
eps_min = 0.01
learning_rate = 0.1
eps_dec = 1e-5
q_table = np.zeros((env.observation_space.n, env.action_space.n))


def get_action(q):
    if np.random.random() > epsilon:
        act = np.argmax(q)
    else:
        act = np.random.randint(0, env.action_space.n)
    return act


episodes = 10000
scores = []
for episode in tqdm(range(episodes)):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = get_action(q_table[state])
        next_state, reward, done, _ = env.step(action)
        score += reward
        max_future_q = np.max(q_table[next_state])
        current_q = q_table[state, action]
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + gamma * max_future_q)
        q_table[state, action] = new_q
        epsilon -= eps_dec if epsilon > eps_min else eps_min
        state = next_state
    scores.append(score)
avg_score = np.mean(scores[-1000:])
print(f"Average score {avg_score}")
