""" 
Base Environments

BallBeamBaseEnv - Base for all ball & beam environments
"""

import time, gym
import numpy as np
from gym import spaces
from gym.utils import EzPickle
from ballbeam_gym.ballbeam import BallBeam

class BallBeamBaseEnv(gym.Env, EzPickle):

    def __init__(self, timestep=None, max_timesteps=None, unit_conversion=None, beam_length=None, ball_radius=None, max_angle=None, max_ang_a=None, init_velocity=None, setpoint=None, reward_scale=None, random_set=None, random_init_vel=None, sleep=None):

        EzPickle.__init__(self)
        self.timestep = timestep
        self.max_timesteps = max_timesteps
        self.reward_scale = reward_scale
        self.random_set = random_set
        self.random_init_vel = random_init_vel
        if random_init_vel and init_velocity is None:  # currently an abirtrary number from 0 to 1, different from reset()
            init_velocity = np.random.random() 
        
        self.action_space = spaces.Box(low=np.array([-max_angle], dtype=np.float32), high=np.array([max_angle], dtype=np.float32))
        self.bb = BallBeam(timestep=timestep,
                           unit_conversion=unit_conversion,
                           beam_length=beam_length,
                           ball_radius=ball_radius,
                           max_angle=max_angle,
                           max_ang_a=max_ang_a,
                           init_velocity=init_velocity,
                           setpoint=setpoint,
                           sleep=sleep,
                           )

        self.last_sleep = time.time()
        self.current_step = 0

    def _sleep_timestep(self):
        """ 
        Sleep to sync cycles to one timestep for rendering by 
        removing time spent on processing.
        """
        duration = time.time() - self.last_sleep
        if not duration > self.timestep:
            time.sleep(self.timestep - duration)
        self.last_sleep = time.time()

    def step(self):
        """
        Update environment for one action
        """
        self.current_step +=1

    def reset(self):
        self.current_step = 0

        setpoint = np.random.uniform(-0.4, 0.4) if self.random_set else None
        init_velocity = np.random.uniform(0, self.bb.max_v/3) if self.random_init_vel else None
        # ^ different from initialization random_init_vel, now an arbitrary number ranging up to max_v/3

        self.bb.reset(setpoint=setpoint, init_velocity=init_velocity)

    def render(self, rc=None, button_info=None):
        """
        Render a timestep and sleep correct time
        """
        self.bb.render(rc=rc, button_info=button_info)
        self._sleep_timestep()


    @property
    def done(self):
        """
        Environment has run a full episode duration
        """
        if self.max_timesteps is None:
            done = not self.bb.on_beam
        else:
            done = self.current_step + 1 >= self.max_timesteps or not self.bb.on_beam

        return done
