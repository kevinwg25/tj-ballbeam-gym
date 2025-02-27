import numpy as np
from math import exp
from gym.spaces import Box
from ballbeam_gym.envs.base import BallBeamBaseEnv


class BallBeamEnv(BallBeamBaseEnv):
    """ BallBeamEnv

    Setpoint environment with a state consisting of key variables

    Parameters
    ----------
    time_step : time of one simulation step, float (s)
    beam_length : length of beam, float (units)
    max_angle : max of abs(angle), float (rads) 
    max_ang_a 
    max_ang_v
    init_velocity : initial speed of ball, float (units/s)
    max_timesteps : maximum length of an episode, int
    action_mode : action space, str ['continuous', 'discrete']
    setpoint : target position of ball, float (units)
    reward_scale : list of weights for ang, pos, vel, accel rewards
    """

    def __init__(self, timestep=0.05, max_timesteps=100, unit_conversion=100, beam_length=100, ball_radius=5, max_angle=0.2, max_ang_v=0.2, init_velocity=0, setpoint=0, startpoint=0, reward_scale=[1, 1, 1, 1], random_set=True, random_init_vel=True, sleep=1, action_mode='discrete'):
       
        kwargs = {'timestep': timestep,
                  'max_timesteps': max_timesteps,
                  'unit_conversion': unit_conversion,
                  'beam_length': beam_length,
                  'ball_radius': ball_radius,
                  'max_angle': max_angle,
                  'max_ang_v': max_ang_v, # rad/s, given by 60 degrees / 0.15 s -> (pi/3 rad)/(0.15 s) = ~7 rad/s. but too high for purposes of training, so use max_ang_v = 0.2
                  'init_velocity': init_velocity,
                  'setpoint': setpoint,
                  'startpoint': startpoint,
                  'reward_scale': reward_scale,
                  'random_set': random_set,
                  'random_init_vel': random_init_vel,
                  'sleep': sleep,
                  'action_mode': action_mode,
                  }

        super().__init__(**kwargs)

        self.observation_space = Box(low=np.array([-max_angle, -np.inf, -np.inf, -beam_length/2], dtype=np.float32),
                                     high=np.array([max_angle, np.inf, np.inf, beam_length/2], dtype=np.float32))

    def sign(self, a, b):
        """
        if only one is zero = bad

        """
        if a==0 and b!=0 or a!=0 and b==0:
            return -1
        return 1 if a*b>0 else -1

    def xor_plus(self, a, b):
        """
        (zero):
            one 0 = bad, both 0 = good
        (nonzero):
            same sign = bad, diff = good -
            = xor
        """
        if a==0 and b==0:
            return 1
        if (a==0 and b!=0) or (b!=0 and b==0):
            return -1
        return 1 if int(a)^int(b) else -1

    def same_sign(self, a, b):
        """
        only one is zero = bad

        """
        if a == 0 and b != 0 or a != 0 and b == 0:
            return False
        return a*b > 0

    def opp_sign(self, a, b):
        if a == 0 and b == 0:
            return True
            # One zero and one nonzero is bad
        if (a == 0 and b != 0) or (a != 0 and b == 0):
            return False
            # Both nonzero: check signs
        return (a > 0) != (b > 0)

    # def call_reward(self, name_of_reward_file):
    #     from name_of_reward_file import reward
    #     # use this reward fucntoin
    #     # otherwise, use defualt
        

    def reward(self):
        dist_k, vel_k, ang_k, aa_k = self.reward_scale
        reward = 0

        # what if dist_set = 0, how take care of that?

        # Distance from setpoint (normalized and squared for quadratic penalty)
        dist_to_set = self.bb.setpoint - self.bb.x
        dist = dist_to_set**2/self.bb.L**2

        reward += (dist_scaled_rew := dist_k * (dist_reward := (1 - 2 * dist)))

        # Velocity near setpoint (normalized)
        vel = self.bb.v**2/self.bb.max_v**2
        vel_sign = self.opp_sign(self.bb.v, dist_to_set)
        if vel_sign:
            reward += (vel_scaled_rew := vel_k * (vel_reward := (vel * dist + (1-vel)*(1-dist))))
        else:
            reward += (vel_scaled_rew := vel_k * (vel_reward := (1-vel)**2 * (1-dist)**2))

        # Beam angle near zero (normalized)
        angle = self.bb.theta**2/self.bb.max_angle**2
        ang_sign = self.same_sign(self.bb.theta, dist_to_set)
        if ang_sign:
            reward += (ang_scaled_rew := ang_k * (angle_reward := (angle * dist + (1-angle)*(1-dist))))
        else:
            reward += (ang_scaled_rew := ang_k * (angle_reward := (1-angle)**2 * (1-dist)**2))

        # Angular acceleration penalty (normalized)

        ang_acc_reward = 0
        aa_s = 0

        # Aggregate reward components and normalize by the sum of weights
        reward /= sum(self.reward_scale)
        components = {
            "x": self.bb.x,
            "v": self.bb.v,
            "theta": self.bb.theta,
            "aa": self.bb.ang_a,
            "a": self.bb.a,
            "dist": dist_reward,
            "vel": vel_reward,
            "ang": angle_reward,
            "ang_acc": ang_acc_reward,
            "dist_s": dist_scaled_rew,
            "vel_s": vel_scaled_rew,
            "ang_s": ang_scaled_rew,
            "ang_acc_s": aa_s,
            "final": reward,
        }

        return reward, components

    def my_reward(self):
        d, ve, ang, aa = self.reward_scale

        # Domain adjustments for exponential scaling
        x_shift = 3  # Horizontal shift for exponential normalization
        x_scale = 1  # Scale factor for exponential decay

        reward = 0

        # what if dist_set = 0, how take care of that?

        # Distance from setpoint (normalized and squared for quadratic penalty)
        dist_to_set = self.bb.setpoint - self.bb.x
        dist = dist_to_set**2/self.bb.L**2
        
        reward += (dist_s := d * (dist_reward := (1 - 2 * dist)))

        # Exponential factor for velocity, angle, and angular acceleration
        exp_factor = exp(-(dist * x_scale - x_shift)) / exp(x_shift)

        # Velocity near setpoint (normalized)
        vel = self.bb.v**2/self.bb.max_v**2
        vel_sign = self.sign(self.bb.v, dist_to_set)
        reward += (vel_s := vel_sign * ve * (vel_reward := (vel * dist + (1-vel)*(1-dist))))

        # Beam angle near zero (normalized)
        angle = self.bb.theta**2/self.bb.max_angle**2
        ang_sign = self.xor_plus(self.bb.theta, dist_to_set)
        reward += (ang_s := ang_sign * ang * (angle_reward := 1 - angle * exp_factor))
        
        # Angular acceleration penalty (normalized)
        ang_acc_norm = self.bb.ang_a**2/self.bb.max_ang_a**2
        aa_sign = 1 if int(self.bb.theta)^int(self.bb.ang_a) else -1
        # reward.append(aa * (ang_acc_reward := 1 - ang_acc_norm * exp_factor))
        reward += (aa_s := aa_sign * aa * (ang_acc_reward := (ang_acc_norm * dist + (1-ang_acc_norm)*(1-dist))))

        # Aggregate reward components and normalize by the sum of weights
        reward /= sum(self.reward_scale)
        components = {
            "x": self.bb.x,
            "v": self.bb.v,
            "theta": self.bb.theta,
            "aa": self.bb.ang_a,
            "a": self.bb.a,
            "dist": dist_reward,
            "vel": vel_reward,
            "ang": angle_reward,
            "ang_acc": ang_acc_reward,
            "dist_s": dist_s,
            "vel_s": vel_s,
            "ang_s": ang_s,
            "ang_acc_s": aa_s,
            "final": reward,

        }

        return reward, components

    def old_reward(self):
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
        ction [continuous] : set angle, float (rad)
        action [discrete] : decrease/keep/increase angle, int [0, 1, 2]
        """
        super().step()
        self.bb.update(self._action_conversion(action))
        state = np.array([float(self.bb.theta), float(self.bb.x), float(self.bb.v), float(self.bb.setpoint)])
        reward, reward_components = self.reward()
        return state, reward, self.done, reward_components
        
    def reset(self):
        """ 
        Reset environment

        Returns
        -------
        observation : simulation state, np.ndarray (state variables)
        """
        super().reset()
       
        return np.array([self.bb.theta, self.bb.x, self.bb.v, self.bb.setpoint])

