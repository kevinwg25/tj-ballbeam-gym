import gym
import ballbeam_gym

# Create environment
env = gym.make('BallBeamSetpoint-v0')
observation = env.reset()

# Introduce a small tilt by adjusting initial conditions (e.g., position or force)
# This applies a small force to simulate tilt.
env.state = [0.1, 0.0, 0.0, -0.1]  # Adjust these values to simulate the tilt

# Run the simulation
for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # Choose an action (random for now)
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()
