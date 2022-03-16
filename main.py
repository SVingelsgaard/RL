#imports
from concurrent.futures import process
from re import S
import gym
import numpy as np
import time
import pandas as pd

import tensorflow as tf
import keras
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

#ML envirement
class Env(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=2,
            shape=(),
            dtype=np.float32
        )

    def reset(self):
        
        return
    def step(self, action):

        info = {}
        #print(sim.state)
        return #sim.state, sim.reward, sim.done, info
    def render(self):
        pass

#sim class. this is a water tank
class Process():
    def __init__(self):
        #params
        self.volume = 0#liters
        self.pv = 0 #liters
        self.sp = 50 #liters
        self.u = False #on off. pump

        #variables
        self.waterOut = 0 #l/s
        self.waterIn = 0 #l/s
        self.e = 0
        self.state = np.array([0], dtype = np.float32)
        self.action = np.array([0], dtype = np.bool)
        self.df = pd.DataFrame()

        #system
        self.timeLast = 0
        self.time = 0 
        self.mafsTime = 0

    def step(self):
        self.controller()
        self.mafs()

    def mafs(self):
        #mafstime
        self.time = time.time()#set time to actual time
        if (self.timeLast == 0):#on first cycle
            self.timeLast = self.time-.2
        self.mafsTime = self.time - self.timeLast #calc mafstime. basically cycletime
        self.timeLast = self.time#uptdate last time

        #innløp
        self.waterIn = int(self.u) * (10 * self.mafsTime)#10 l/s

        #utløp
        self.waterOut = 1 * self.mafsTime

        #calc pv
        self.pv += self.waterIn - self.waterOut
        
        
        self.state = np.array([self.pv])

    def reset(self):
        self.pv = 0

        self.state = np.array([self.pv])

    
    def controller(self):
        self.e = self.sp - self.pv
        if (self.e > 0):
            self.u = True
        else:
            self.u = False

if __name__ == '__main__':
    sim = Process()

    env = Env()

    for i in range(1000):
        time.sleep(.02)
        sim.step()
        print(sim.state)



        
        

