import gym
import ballbeam_gym
import numpy as np
from math import pi

kwargs = {
    'setpoint': 0.3,
    # 'reward_scale': reward_scale,
    'random_set': False,
    'random_init_vel': False,
    'beam_length': 100,
    'ball_radius': 5,
}
env = gym.make('BallBeam-v0', **kwargs)

# position should be weighted 2-3x higher than velocity

Kp = 2
Kv = 1

for i in range(1000):   
    theta = np.array(val := Kp*(env.bb.x - env.bb.setpoint) + Kv*(env.bb.v))
    # print(val*180/pi)
    _, reward, done, rc = env.step(theta)
    env.render(rc)
    if done:
        env.reset()