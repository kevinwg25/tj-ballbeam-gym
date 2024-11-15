import os, gym
from datetime import datetime
from PPO import PPO
import matplotlib.pyplot as plt
import ballbeam_gym

show_info = False
show_graph = False
rend = False
save_logs = False
setpoint = 0.3
reward_scale = [3, 5, 1, 1]  # dist, vel, angle, ang_accel

kwargs = {
    'setpoint': setpoint,
    'reward_scale': reward_scale,
    'random_set': False,
    'random_init_vel': False,
    'beam_length': 100,
    'ball_radius': 5,
}

################################### Training ###################################
def train():
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
    env_name = "BallBeam-v0"
    env = gym.make(env_name, **kwargs)
    fn = input("Enter filename to train, blank for default, or a number for checkpoint (-1 for latest): ")
    rs = "-".join(str(float(i)) for i in reward_scale)
    weights_directory = "trained" + "/"# + env_name + "/"
    if fn.lstrip("-").isnumeric():
        if int(fn)>=0:
            fn = "PPO_{}_{}_{}.pth".format(env_name, rs)
        else:
            files = os.listdir(weights_directory)
            check = []
            for fn in files:
                if rs in fn:
                    point = fn.split(env_name+"_")[-1][:-4].split("_")[-1]
                    if point.isnumeric():
                        check.append(int(point))
            n = max(check)+1 if check else 0
            fn = "PPO_{}_{}_{}.pth".format(env_name, rs, n)
    elif len(fn)==0:
        fn = "PPO_{}_{}.pth".format(env_name, rs)
    
    max_ep_len = 1000                   # max timesteps in one episode
    max_epochs = 1000
    max_training_timesteps = int(max_epochs*max_ep_len)   # break training loop if timeteps > max_training_timesteps
    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    lr_actor = 0.003       # learning rate for actor network
    lr_critic = 0.01       # learning rate for critic network
    #####################################################

    if show_info:
        print("training environment name : " + env_name)

    # state space dimens1ion
    state_dim = env.observation_space.shape[0]

    # action space dimension
    action_dim = env.action_space.shape[0]

    ###################### logging ######################
    if save_logs:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        run_num = len(os.listdir(log_dir))

        log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    ################### checkpointing ###################

    if not os.path.exists(weights_directory):
          os.makedirs(weights_directory)
    
    checkpoint_path = weights_directory + fn
    
    #####################################################

    ####### initialize environment hyperparameters ######
    if show_graph:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Average Episode Reward')
        ax.set_title('Training Progress - PPO')
        reward_line, = ax.plot([], [], lw=2)

    ################# training procedure ################

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    start_time = datetime.now().replace(microsecond=0)
    print()
    print("Saving weights file to {}".format(fn))
    print()
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")
    if save_logs:
        log_f = open(log_f_name,"w+")
        log_f.write('episode,timestep,reward\n')

    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    epoch = 0
    avg_rewards = []
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0
        for states in range(1, max_ep_len+1):
            info = env.get_info()
            # select action with policy
            action = ppo_agent.select_action(state, info)
            state, reward, done, _ = env.step(action)
            if rend:
                env.render()

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # decay action std of ouput action distribution
            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if save_logs and time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

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
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
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
    if save_logs:
        log_f.close()
    env.close()

    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
