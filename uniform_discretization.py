import gymnasium as gym
import numpy as np

# Configuración del entorno
env = gym.make("MountainCar-v0")

# Parámetros de aprendizaje
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 0.1  # Parámetro epsilon-greedy
num_episodes = 1000  # Número de episodios

# Configuración de la discretización
num_bins_position = 20  # Número de divisiones para la posición
num_bins_velocity = 20  # Número de divisiones para la velocidad
bins_position = np.linspace(-1.2, 0.6, num_bins_position - 1)  # Bordes de las celdas
bins_velocity = np.linspace(-0.07, 0.07, num_bins_velocity - 1)  # Bordes de las celdas

# Crear la tabla Q inicial
Q = np.zeros((num_bins_position, num_bins_velocity, env.action_space.n))

# Función para discretizar estados continuos
def discretize_state(state):
    position, velocity = state
    position_idx = np.digitize(position, bins_position)
    velocity_idx = np.digitize(velocity, bins_velocity)
    return position_idx, velocity_idx

# Función para seleccionar una acción (epsilon-greedy)
def select_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    position_idx, velocity_idx = discretize_state(state)
    return np.argmax(Q[position_idx, velocity_idx])

# Entrenamiento con Q-Learning
for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        position_idx, velocity_idx = discretize_state(state)
        action = select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        next_position_idx, next_velocity_idx = discretize_state(next_state)
        total_reward += reward

        # Actualización de Q-Learning
        current_q = Q[position_idx, velocity_idx, action]
        max_next_q = np.max(Q[next_position_idx, next_velocity_idx])
        Q[position_idx, velocity_idx, action] = current_q + alpha * (
            reward + gamma * max_next_q - current_q
        )

        state = next_state

    print(f"Episodio {episode + 1}: Recompensa total = {total_reward}")

# Prueba del agente entrenado
epsilon = 0  # Deshabilitar exploración
for i in (range(5)):
    env = gym.make("MountainCar-v0", render_mode="human")
    state = env.reset()[0]
    done = False
    while not done:
        env.render()
        action = select_action(state)
        state, _, done, _, _ = env.step(action)

    env.close()