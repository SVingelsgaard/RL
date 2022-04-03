import numpy as np
import pandas as pd
import tensorflow as tf
import time
import pickle


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
        self.y = []
        with open('data.txt', 'w') as f:
                f.write('')
                print("wrote data")

        #system
        self.timeLast = 0
        self.time = 0 
        self.mafsTime = 0
        self.runTime = 0

    def step(self):
        #self.controller()
        self.mafs()

        #reward
        self.reward = 0
        if (self.e < 3) and (self.e > -3):
            self.reward = 1  

        self.score += self.reward

        if self.runTime > 10:
            with open('data.txt', 'a') as f:
                f.write(str(self.y))
                f.write('\n')
                print("wrote data")
            self.y = []
            self.done = True

        self.plot()

    def mafs(self):
        #mafstime
        self.time = time.time()#set time to actual time
        if (self.timeLast == 0):#on first cycle
            self.timeLast = self.time-.2
        self.mafsTime = self.time - self.timeLast #calc mafstime. basically cycletime
        self.timeLast = self.time#uptdate last time
        self.runTime += self.mafsTime#update runtime

        #assign action
        self.u = self.action

        #innløp
        self.waterIn = int(self.u) * (10)# * self.mafsTime)#10 l/s

        #utløp
        self.waterOut = 1 #* self.mafsTime

        #calc pv
        self.pv += self.waterIn - self.waterOut
        self.e = self.sp - self.pv
        
        self.state = np.array([self.pv])


    def reset(self):
        self.pv = 0
        self.reward = 0
        self.runTime = 0 
        self.score = 0 
        self.done = False

        self.state = np.array([self.pv])

    
    def controller(self):
        if (self.e > 0):
            self.u = True
        else:
            self.u = False
    
    def plot(self):
        self.y.append(int(self.state)) 

