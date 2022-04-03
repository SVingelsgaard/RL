#imports
from concurrent.futures import process
from re import S
from sre_parse import State
import gym
import gym_game
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

if __name__ == '__main__':
    #sim = Process()

    env = gym.make('TempSim-v0')

    model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(1,1,)),#wrong shape or sum
                tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
                tf.keras.layers.Dense(2, activation=tf.nn.softmax)#maby not right...
                ])
    agent = DQNAgent(
            
                model=model, 
                memory=SequentialMemory(limit=10000, window_length=1), 
                policy=BoltzmannQPolicy
                (), 

                nb_actions=2, 
                nb_steps_warmup=10,
                target_model_update=1e-2
                )


    agent.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])

    agent.fit(env, nb_steps=10000, visualize=False, verbose=1)
    #sim.agent.save_weights('dqn_weights.h5f', overwrite=True)

    #sim.agent.load_weights('dqn_weights.h5f')
    agent.test(env, nb_episodes=1, visualize=False)

    




        
        

