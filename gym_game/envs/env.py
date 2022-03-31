import gym
import gym_game
import numpy as np
from gym_game.envs.process import Process
class Env(gym.Env):
    def __init__(self):
        self.sim = Process()
        self.action_space = gym.spaces.Discrete(2)
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=100,
            shape=([1,1,]),
            dtype=np.float32
        )

    def reset(self):
        self.sim.reset()
        return self.sim.state
    def step(self, action):
        self.sim.action = action

        self.sim.step()

        info = {}
        #print(sim.state, sim.reward)

        return self.sim.state, self.sim.reward, self.sim.done, info
    def render(self):
        pass