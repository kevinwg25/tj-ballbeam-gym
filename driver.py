from BallBeamModel import BallBeamModel

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

args = {
    'setpoint': 0.3,
    'reward_scale': [10, 3, 1, 1],
    'random_set': False,
    'random_init_vel': False,
    # 'max_ang_a' : 0.6 # changing this doesnt actually cap the acceleration with how our code is written now. changing this just changes the reward
}

bbm = BallBeamModel(args=args)

bbm.test(frame_delay=1, total_test_episodes=100, fn='temp.pth')
# bbm.train()