from BallBeamModel import BallBeamModel

args = {
    'setpoint': 0.3,
    'reward_scale': [10, 3, 1, 1],
    'random_set': False,
    'random_init_vel': False,
    # 'max_ang_a' : 0.6 # changing this doesnt actually cap the acceleration with how our code is written now. changing this just changes the reward
}

bbm = BallBeamModel(args=args)

fd = 1

# bbm.test(frame_delay=fd, total_test_episodes=100, fn='temp')
# bbm.train()
bbm.run_pid(frame_delay=fd)