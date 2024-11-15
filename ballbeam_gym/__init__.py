from gym.envs.registration import register

__version__ = '0.0.5'

register(
    id='BallBeam-v0',
    entry_point='ballbeam_gym.envs:BallBeamEnv',
)