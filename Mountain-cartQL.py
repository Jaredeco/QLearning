import gym
import numpy as np
from tqdm import tqdm

env = gym.make('MountainCar-v0')
gamma = 0.95
epsilon = 1
eps_min = 0.2
learning_rate = 0.1
eps_dec = 1e-2
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
q_table = np.zeros((DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


def get_action(q):
    if np.random.random() > epsilon:
        act = np.argmax(q)
    else:
        act = np.random.randint(0, env.action_space.n)
    return act


episodes = 3000
scores = []
for episode in tqdm(range(episodes)):
    state = get_discrete_state(env.reset())
    done = False
    score = 0
    while not done:
        action = get_action(q_table[state])
        next_state, reward, done, _ = env.step(action)
        next_dis_state = get_discrete_state(next_state)
        score += reward
        if not done:
            max_future_q = np.max(q_table[next_dis_state])
            current_q = q_table[state + (action,)]
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + gamma * max_future_q)
            q_table[state + (action,)] = new_q
        elif next_state[0] >= env.goal_position:
            q_table[state + (action,)] = 0
        epsilon -= eps_dec if epsilon > eps_min else eps_min
        state = next_dis_state
    scores.append(score)
avg_score = np.mean(scores)
print(f"Average score {avg_score}")
