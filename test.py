import time, gym, os
from PPO import PPO
import ballbeam_gym

reward_scale = [3, 5, 1, 1]

kwargs = {
    'setpoint': 0.3,
    'reward_scale': reward_scale,
    'random_set': False,
    'random_init_vel': False,
    'beam_length': 100,
    'ball_radius': 5,
    'sleep': None, # integer for sleep, None for screenshot
}
frame_delay = 0
total_test_episodes = 100

def test():
    env_name = "BallBeam-v0"
    weights_directory = "trained" + "/"# + env_name + "/"
    rs = "-".join(str(float(i)) for i in reward_scale)

    fn = input("Enter filename to test, blank for default, a number for checkpoint (-1 for latest), or comma-separated list of weights: ")
    if fn.lstrip("-").isnumeric():
        if int(fn) >= 0:
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
    elif len(fn) == 0:
        fn = "PPO_{}_{}.pth".format(env_name, rs)
    elif fn.count(",") == 3:  # hardcoded number of weights = 4
        file_name = "PPO_{}_{}.pth".format(env_name, "-".join(fn.split(", ")))
    print("============================================================================================")

    ################## hyperparameters ##################
    max_ep_len = 10000
    action_std = 0.1
    render = True              # render environment on screen
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################
    
    env = gym.make(env_name, **kwargs)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    checkpoint_path = weights_directory + fn
    print("loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

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
            if done:
                break
        ppo_agent.buffer.clear()
        test_running_reward += (avg_reward := ep_reward/frameNum)
        print('Episode: {} \t\t\t Avg Reward: {}'.format(ep, round(avg_reward, 3)))

    env.close()

    print("============================================================================================")
    print("average test reward:", round(test_running_reward/total_test_episodes, 2))
    print("============================================================================================")


if __name__ == '__main__':
    test()
