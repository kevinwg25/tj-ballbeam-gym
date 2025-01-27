from BallBeamModel import BallBeamModel

args = {
    'setpoint': 0,
    'reward_scale': [10, 3, 1, 1],
    'random_set': False,
    'random_init_vel': False,
}

bbm = BallBeamModel(args=args)
fd = 0
inp = input("Enter for train, 1 for test, 2 for run_pid: ")

# to test, press 1, and then press enter
# it will load defalt file as described by reward_scale in args above

if not inp:
    bbm.train(show_graph=False)
if inp == "1":
    bbm.test(frame_delay=fd, total_test_episodes=10, max_ep_len=1000000)
elif inp == "2":
    bbm.run_pid(frame_delay=fd)