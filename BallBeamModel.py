import time, gym, os
from PPO import PPO
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import ballbeam_gym

class BallBeamModel:
    def __init__(self, args={}, K_epochs=80, eps_clip=0.2, gamma=0.99, lr_actor=0.003, lr_critic=0.01, action_std_decay_rate=0.05, min_action_std=0.1, action_std_decay_freq=int(2.5e5), max_epochs=1000, save_logs=False):
        """
        args parameters

        kwargs : for setpoint
        K_epochs : 80
        eps_clip : 0.2
        gamma : 0.99
        lr_actor : 0.003
        lr_critic : 0.01
        action_std_decay_rate : 0.05
        min_action_std : 0.1
        action_std_decay_freq : int(2.5e5)
        max_epochs : 1000
        show_graph : False, learning curve
        save_logs : False, in a csv file
        """
        self.env_name = "BallBeam-v0"
        self.state_dim = 4
        self.action_dim = 1
        if "action_mode" in args and args["action_mode"] == 'continuous':
            self.has_continuous_action_space = True
            self.action_dim = 1
        else:
            self.has_continuous_action_space = False
            self.action_dim = 3 # hardcoded number of actions
        self.weights_dir = "trained/"

        """
        Setpoint class parameters

        timestep: default 0.05 s
        max_timesteps: default 100
        beam_length: default 1.0
        ball_radius: default 0.05
        max_angle: default 0.2 rad
        max_ang_a: default 26.18 rad/s^2 (from 0.2 sec / 60 deg)
        init_velocity: default 0 m/s
        setpoint: fraction of beam length, default 0
        reward_scale: default [1,1,1,1]
        random_set: default True
        random_init_vel: default True

        """
        self.kwargs = args
        
        self.K_epochs = K_epochs        # update policy for K epochs
        self.eps_clip = eps_clip        # clip parameter for PPO
        self.gamma = gamma              # discount factor
        self.lr_actor = lr_actor        # learning rate for actor
        self.lr_critic = lr_critic      # learning rate for critic

        self.action_std_TEST = 0.1
        self.max_ep_len_TEST = 10000
        
        self.action_std_TRAIN = 0.6
        self.max_ep_len_TRAIN = 10000
        
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std
        self.action_std_decay_freq = action_std_decay_freq
        self.max_epochs = max_epochs
        self.update_timestep = self.max_ep_len_TRAIN * 4
        self.max_training_timesteps = int(self.max_epochs*self.max_ep_len_TRAIN)
        self.print_freq = self.max_ep_len_TRAIN
        self.log_freq = self.max_ep_len_TRAIN * 2
        self.save_model_freq = int(1e5)
        self.save_logs = save_logs
        self.block = '-'*92
        self.big_block = '='*92

    def get_fn(self):
        scale = self.kwargs["reward_scale"] if "reward_scale" in self.kwargs else [1,1,1,1] # hardcoded number of weights (subject to change?)
        rs = "-".join(str(float(i)) for i in scale)
        text = 'Enter filename, blank for default, number for checkpoint (-1 for latest), or comma (", ") separated integer weights: '
        fn = input(text)
        if fn.lstrip("-").isnumeric():
            if (n := int(fn)) >= 0:
                fn = "PPO_{}_{}_{}.pth".format(self.env_name, rs, n)
            else:
                files = os.listdir(self.weights_dir)
                max_n = -1
                for fn in files:
                    if rs in fn:
                        point = fn.split(self.env_name+"_")[-1][:-4].split("_")[-1]
                        if point.isnumeric() and (n := int(point))>max_n:
                            max_n = n
                n = max_n+1
                fn = "PPO_{}_{}_{}.pth".format(self.env_name, rs, n)
        elif len(fn) == 0:
            fn = "PPO_{}_{}.pth".format(self.env_name, rs)
        elif fn.count(",") == 3:  # hardcoded number of weights = 4
            fn = "PPO_{}_{}.pth".format(self.env_name, "-".join(fn.split(", ")))
        else:
            fn+=".pth"
        return fn

    def train(self, show_graph=True):
        fn = self.get_fn()

        #  call function for opening tkinter window
        # name_of_reward_func = functoin_wascalled()
        # kwarsg['reward_func'] = name_of_reward_func
        env = gym.make(self.env_name, **self.kwargs)
        if self.save_logs:
            log_dir = "logs/" + self.env_name
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            run_num = len(os.listdir(log_dir))
            log_fn = "{}/PPO_{}_log_{}.csv".format(log_dir, self.env_name, str(run_num))
            log_f = open(log_fn, "w+")
            log_f.write('episode,timestep,reward\n')
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        if show_graph:
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Average Episode Reward')
            ax.set_title('Training Progress - PPO')
            reward_line, = ax.plot([], [], lw=2)
        checkpoint_path = self.weights_dir + fn
        print(f"\nSaving weights file to {checkpoint_path}\nStarted training at (GMT) : {(start_time := datetime.now().replace(microsecond=0))}")
        print(self.block)
        ppo_agent = PPO(self.state_dim, self.action_dim, self.lr_actor, self.lr_critic, self.gamma, self.K_epochs, self.eps_clip, self.has_continuous_action_space, self.action_std_TRAIN)
        print_running_reward = print_running_episodes = log_running_reward = log_running_episodes = time_step = i_episode = epoch = 0
        avg_rewards = []
        
        
        
        while time_step <= self.max_training_timesteps:
            state = env.reset()
            current_ep_reward = 0
            for states in range(1, self.max_ep_len_TRAIN+1):
                # select action with policy
                action = ppo_agent.select_action(state)
                state, reward, done, _ = env.step(action)

                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                # update PPO agent
                if time_step % self.update_timestep == 0:
                    ppo_agent.update()

                # decay action std of ouput action distribution
                if self.has_continuous_action_space and time_step % self.action_std_decay_freq == 0:
                    ppo_agent.decay_action_std(self.action_std_decay_rate, self.min_action_std)

                # log in logging file
                if self.save_logs and time_step % self.log_freq == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)
                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()
                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % self.print_freq == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    avg_rewards.append(print_avg_reward)
                    print_avg_reward = round(print_avg_reward, 2)
                    print("Episode : {} \t\t\t Timestep : {} \t\t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                    print_running_reward = 0
                    print_running_episodes = 0
                    epoch += 1
                    if show_graph:
                        reward_line.set_data(range(len(avg_rewards)), avg_rewards)
                        ax.set_xlim(0, epoch)
                        ax.set_ylim(0, 1.1*max(avg_rewards))
                        plt.pause(0.1)

                # save model weights
                if time_step % self.save_model_freq == 0:
                    print(self.block)
                    ppo_agent.save(checkpoint_path)
                    print("saved model at : " + checkpoint_path)
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print(self.block)

                # stop if episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1
            i_episode += 1
        if show_graph:
            plt.ioff()
            plt.show()
        if self.save_logs:
            log_f.close()
        
        env.close()
        print(self.big_block)
        print()
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time := datetime.now().replace(microsecond=0))
        print("Total training time  : ", end_time - start_time)
    
    def test(self, frame_delay=0, max_ep_len=1000, total_test_episodes=10, fn=None, render=True):
        env = gym.make(self.env_name, **self.kwargs)
        ppo_agent = PPO(self.state_dim, self.action_dim, self.lr_actor, self.lr_critic, self.gamma, self.K_epochs, self.eps_clip, self.has_continuous_action_space, self.action_std_TEST)
        fn = self.get_fn() if fn is None else fn+".pth"
        print(self.big_block)
        checkpoint_path = self.weights_dir + fn
        ppo_agent.load(checkpoint_path)
        print(f"Loaded weights from: {fn}")
        print(self.big_block)
        test_running_reward = 0
        reward_components = True
        for ep in range(1, total_test_episodes+1):
            ep_reward = 0
            state = env.reset()
            for frameNum in range(1, max_ep_len+1):
                if render:
                    env.render(rc=reward_components, button_info=(ep, frameNum))
                    time.sleep(frame_delay)
                action = ppo_agent.select_action(state)
                state, reward, done, reward_components = env.step(action)
                ep_reward += reward
                ppo_agent.buffer.clear()
                test_running_reward += (avg_reward := ep_reward/frameNum)
                if done:
                    break
            print('Episode: {} \t\t\t Avg Reward: {}'.format(ep, round(avg_reward, 3)))
        
        env.close()
        print(self.big_block)
        print()
        print("Average test reward:", round(test_running_reward/total_test_episodes, 2))

    def run_pid(self, frame_delay=0, Kp=2, Kv=1, print_angle=False, max_ep_len=1000):
        env = gym.make(self.env_name, **self.kwargs)
        for i in range(max_ep_len):
            theta = np.array(val := Kp*(env.bb.x - env.bb.setpoint) + Kv*(env.bb.v))
            if print_angle:
                print(val * 180 / 3.14159)
            _, _, done, rc = env.step(theta)
            env.render(rc)
            time.sleep(frame_delay)
            if done:
                state = env.reset() 

    def do_nothing(self, frame_delay=0, max_ep_len=1000):
        env = gym.make(self.env_name, **self.kwargs)
        state = env.reset()
        for i in range(max_ep_len):
            theta = 40  # No action, keep the beam stationary
            _, _, done, rc = env.step(theta) 
            env.render(rc) 
            time.sleep(frame_delay)
            if done:
                state = env.reset() 



