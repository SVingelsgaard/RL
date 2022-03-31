from gym.envs.registration import register

register(id='TempSim-v0',
         entry_point='gym_game.envs:Env'
         )
