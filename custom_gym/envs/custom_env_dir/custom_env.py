import gym
class CustomEnv(gym.Env):
    def __inti__(self):
        print("Env Initilized")
    def step(self):
        print("Step done")
    def restet(self):
        print("Env reset")