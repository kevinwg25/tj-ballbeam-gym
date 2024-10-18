import gym
import ballbeam_gym
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# Initialize Ball & Beam environment
kwargs = {'timestep': 0.05, 
          'setpoint': 0.0,
          'beam_length': 1.0,
          'max_angle': 0.2,
          'init_velocity': 0.6}

env = gym.make('BallBeamSetpoint-v0', **kwargs)
observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

def create_actor():
    inputs = layers.Input(shape=(observation_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(action_dim, activation='tanh')(x) 
    model = tf.keras.Model(inputs, outputs)
    return model

def create_critic():
    inputs = layers.Input(shape=(observation_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1)(x) 
    model = tf.keras.Model(inputs, outputs)
    return model

actor_model = create_actor()
critic_model = create_critic()

optimizer_actor = Adam(learning_rate=3e-4)
optimizer_critic = Adam(learning_rate=3e-4)

class PPOBuffer:
    def __init__(self, observation_dim, action_dim, size, gamma=0.99, lam=0.95):
        self.observation_buffer = np.zeros((size, observation_dim), dtype=np.float32)
        self.action_buffer = np.zeros((size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.logprob_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, action, reward, value, logprob):
        self.observation_buffer[self.ptr] = obs
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.value_buffer[self.ptr] = value
        self.logprob_buffer[self.ptr] = logprob
        self.ptr += 1

    def finish_path(self, last_value=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)

        self.return_buffer[path_slice] = self.discount_cumsum(rewards, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        advantage_mean, advantage_std = np.mean(self.advantage_buffer), np.std(self.advantage_buffer)
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / (advantage_std + 1e-8)
        return [
            self.observation_buffer, 
            self.action_buffer, 
            self.advantage_buffer, 
            self.return_buffer, 
            self.logprob_buffer
        ]

    @staticmethod
    def discount_cumsum(x, discount):
        return np.array([sum(x[i:] * (discount ** np.arange(len(x[i:])))) for i in range(len(x))])

def ppo_loss(old_log_probs, advantages, actions, new_log_probs):
    ratio = tf.exp(new_log_probs - old_log_probs)
    clipped_ratio = tf.clip_by_value(ratio, 1 - 0.2, 1 + 0.2)  # Clip range [0.8, 1.2]
    loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
    return loss

def critic_loss(returns, values):
    return tf.keras.losses.MeanSquaredError()(returns, values)

def train_ppo(epochs=100, steps_per_epoch=1000):
    buffer = PPOBuffer(observation_dim, action_dim, steps_per_epoch * epochs)
    episode_rewards = []
    avg_rewards = []

    # Initialize the plot window before training starts
    plt.ion() 
    fig, ax = plt.subplots()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Average Episode Reward')
    ax.set_title('Training Progress - PPO')
    reward_line, = ax.plot([], [], lw=2)

    for epoch in range(epochs):
        state = env.reset()
        episode_reward = 0

        # note to future self:
        # steps per epoch would be a better way to store states since it creates a defined
        # size for the buffer. If the for loop ended when the ball fell off the beam then
        # that could take any number of steps, making steps harder to write to the buffer
        # and to access -Kevin

        for step in range(steps_per_epoch):
            env.render()
            state = state.reshape(1, -1)
            value = critic_model(state)
            action = actor_model(state)
            log_prob = tf.math.log(action)
            
            next_state, reward, done, _ = env.step(action.numpy()[0])
            episode_reward += reward

            buffer.store(state, action, reward, value, log_prob)
            state = next_state

            if done:
                buffer.finish_path(last_value=0)
                state = env.reset()
                # episode_reward = episode_reward / steps_per_epoch
                # ^^ change back
                episode_rewards.append(episode_reward)
                episode_reward = 0

        for _ in range(10):
            obs, act, adv, ret, logp = buffer.get()

            with tf.GradientTape() as tape:
                new_probs = actor_model(obs)
                actor_loss = ppo_loss(logp, adv, act, new_probs)
            actor_grads = tape.gradient(actor_loss, actor_model.trainable_variables)
            optimizer_actor.apply_gradients(zip(actor_grads, actor_model.trainable_variables))

            with tf.GradientTape() as tape:
                value = critic_model(obs)
                critic_loss_val = critic_loss(ret, value)
            critic_grads = tape.gradient(critic_loss_val, critic_model.trainable_variables)
            optimizer_critic.apply_gradients(zip(critic_grads, critic_model.trainable_variables))

        avg_reward = np.mean(episode_rewards)
        avg_rewards.append(avg_reward)
        print(f"Epoch {epoch}, Average Episode Reward: {avg_reward}")

        reward_line.set_data(range(len(avg_rewards)), avg_rewards)
        ax.set_xlim(0, epochs) 
        ax.set_ylim(0, max(avg_rewards) + 10) 
        plt.draw() 
        plt.pause(0.1)

    plt.ioff() 
    plt.show() 

train_ppo(epochs=10, steps_per_epoch=1000)


train_ppo()
