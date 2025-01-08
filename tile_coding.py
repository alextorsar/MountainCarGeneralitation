import gymnasium as gym
import numpy as np
from tile_functions import Tile_functions


env = gym.make("MountainCar-v0")


# Configuración de Tile Coding
num_tilesets = 8  # Número de tilesets (superposiciones)
num_tiles_per_dim = 10  # Divisiones por dimensión
iht_size = 4096  # Tamaño del índice de hash (para eficiencia)

alpha = 0.1 / num_tilesets  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 0.1

weights = np.zeros((iht_size, env.action_space.n))

low = [-1.2, -0.07]  # Límites inferiores del espacio de estados (posición, velocidad)
high = [0.6, 0.07]   # Límites superiores del espacio de estados (posición, velocidad)

tile_coder = Tile_functions(num_tilesets=num_tilesets, num_tiles_per_dim=num_tiles_per_dim, low=low, high=high)

def q_value(state, action):
    active_tiles = tile_coder.get_tiles(state)
    return sum(weights[tile, action] for tile in active_tiles)

def select_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    q_values = [q_value(state, action) for action in range(env.action_space.n)]
    return np.argmax(q_values)

num_episodes = 500
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    while not done:
        action = select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_action = max(range(env.action_space.n), key=lambda action: q_value(next_state, action))
        active_tiles = tile_coder.get_tiles(state)
        current_q = q_value(state, action)
        for tile in active_tiles:
            weights[tile, action] += alpha * (reward + gamma * q_value(next_state, next_action) - current_q)
        state = next_state



for i in range(10):
    env = gym.make("MountainCar-v0", render_mode="human")
    epsilon = 0
    state, info = env.reset()
    done = False
    while not done:
        env.render()
        action = select_action(state)
        next_state, _, done, _, _ = env.step(action)
        state = next_state

    env.close()