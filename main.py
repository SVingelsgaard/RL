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
        self.action_space = gym.spaces.Discrete(3)
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=100,
            shape=([1,1,]),
            dtype=np.float32
        )

    def reset(self):
        self.state()
        return sim.state
    def step(self, action):
        sim.action = action

        sim.step()

        info = {}

        return sim.state, sim.reward, sim.done, info
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
        self.reward = 0
        self.done = False
        self.score = 0

        #system
        self.timeLast = 0
        self.time = 0 
        self.mafsTime = 0

        #RL
        self.env = Env()
        #create model
        self.model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(1,1,)),#wrong shape or sum
                tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
                tf.keras.layers.Dense(2, activation=tf.nn.softmax)#maby not right...
                ]) 

        #create agent
        self.dqn = DQNAgent(
            
                model=self.model, 
                memory=SequentialMemory(limit=50000, window_length=1), 
                policy=BoltzmannQPolicy(), 

                nb_actions=2, 
                nb_steps_warmup=10,
                target_model_update=1e-2
                )
        #compile agent
        self.dqn.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])

    def step(self):
        #self.controller()
        self.mafs()

        #reward
        if (self.e < 3) and (self.e > -3):
            self.reward += 1
        self.score += self.reward

    def mafs(self):
        #mafstime
        self.time = time.time()#set time to actual time
        if (self.timeLast == 0):#on first cycle
            self.timeLast = self.time-.2
        self.mafsTime = self.time - self.timeLast #calc mafstime. basically cycletime
        self.timeLast = self.time#uptdate last time

        #assign action
        self.u = self.action

        #innløp
        self.waterIn = int(self.u) * (10 * self.mafsTime)#10 l/s

        #utløp
        self.waterOut = 1 * self.mafsTime

        #calc pv
        self.pv += self.waterIn - self.waterOut
        self.e = self.sp - self.pv
        
        self.state = np.array([self.pv])

    def reset(self):
        self.pv = 0
        self.reward = 0

        self.state = np.array([self.pv])

    
    def controller(self):
        if (self.e > 0):
            self.u = True
        else:
            self.u = False

if __name__ == '__main__':
    sim = Process()

    for i in range(250):
        time.sleep(.02)
        sim.env.action_space.sample()
        print(sim.action)
        sim.step()
        #print(sim.state)



        
        

