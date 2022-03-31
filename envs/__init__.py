from gym.envs.registration import register
from importlib_metadata import entry_points

register(id='CustomEnv-v0',
         entry_point='env.custom_env_dir:CustomEnv'
         )
