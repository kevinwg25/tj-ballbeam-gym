import numpy as np
from math import exp
from gym import spaces
from ballbeam_gym.envs.base import BallBeamBaseEnv
from math import sin

class BallBeamEnv(BallBeamBaseEnv):
    """ BallBeamEnv

    Setpoint environment with a state consisting of key variables

    Parameters
    ----------
    time_step : time of one simulation step, float (s)
    beam_length : length of beam, float (units)
    max_angle : max of abs(angle), float (rads) 
    init_velocity : initial speed of ball, float (units/s)
    max_timesteps : maximum length of an episode, int
    action_mode : action space, str ['continuous', 'discrete']
    setpoint : target position of ball, float (units)
    reward_scale : list of weights for ang, pos, vel, accel rewards
    """

    def __init__(self, timestep=0.05, max_timesteps=100, unit_conversion=100, beam_length=30, ball_radius=1, max_angle=0.2, max_ang_a=26.18, init_velocity=0, setpoint=0, reward_scale=[1, 1, 1, 1], random_set=True, random_init_vel=True, sleep=1):
       
        kwargs = {'timestep': timestep,
                  'max_timesteps': max_timesteps,
                  'unit_conversion': unit_conversion,
                  'beam_length': beam_length,
                  'ball_radius': ball_radius,
                  'max_angle': max_angle,
                  'max_ang_a': max_ang_a,
                  'init_velocity': init_velocity,
                  'setpoint': setpoint,
                  'reward_scale': reward_scale,
                  'random_set': random_set,
                  'random_init_vel': random_init_vel,
                  'sleep': sleep,
                  }

        super().__init__(**kwargs)

        self.observation_space = spaces.Box(
            low=np.array([-max_angle, -np.inf, -np.inf, -beam_length/2], dtype=np.float32),
            high=np.array([max_angle, np.inf, np.inf, beam_length/2], dtype=np.float32))
        
    def reward(self):
        d, ve, ang, aa = self.reward_scale

        # arbitrary domain for exponential function: D = [0, domain_scale]; increase it for higher penalty to larger distances
        # horizontal shift and domain scale for reasonable bounds of e^-x on [0, x_scale]
        x_shift = 3
        x_scale = 1#0
        
        reward = []

        # distance from setpoint
        r = (self.bb.x - self.bb.setpoint)**2/self.bb.L**2
        reward.append(d*(dist := 1 - r))

        # closer to setpoint should be higher penalty
        # dont multiply everything becaues then theres no point?
        normal_exp = exp(-(r*x_scale - x_shift))/exp(x_shift)

        # velocity near setpoint
        v = self.bb.v**2/self.bb.max_v**2
        reward.append(ve*(vel := 1 - v*normal_exp))
        # print(self.bb.v, v, normal_exp, v*normal_exp)
        # print()

        # "setpoint" angle of beam = 0
        a = self.bb.theta**2/self.bb.max_angle**2
        reward.append(ang*(angle := 1 - a*normal_exp))

        # as velocity is higher, angular acceleartion should be lower?

        # angular acceleration - not actually normalized, replace later with torque?
        aa = self.bb.ang_a**2/self.bb.max_ang_a**2
        reward.append(aa*(ang_accel := 1 - aa*normal_exp))

        """
        vel stays too high ~0.94
        aa really too high
        position is suspiciously above 0.5 usually

        ang changes well
        """

        # print([dist, vel, angle, ang_accel])
        # print()
        sum_reward = sum(reward) / sum(self.reward_scale)
        components = {"dist": dist, "vel": vel, "ang": angle, "ang_acc": ang_accel}

        return sum_reward, components
    
        
    def step(self, action):
        """
        Update environment for one action

        Parameters
        ----------
        action : float, the angle in (rad)
        """
        super().step()

        self.bb.update(action)
        obs = np.array([float(self.bb.theta), float(self.bb.x), float(self.bb.v), float(self.bb.setpoint)])
        rew, rew_components = self.reward()
        return obs, rew, self.done, rew_components
        
    def reset(self):
        """ 
        Reset environment

        Returns
        -------
        observation : simulation state, np.ndarray (state variables)
        """
        super().reset()
       
        return np.array([self.bb.theta, self.bb.x, self.bb.v, self.bb.setpoint])

