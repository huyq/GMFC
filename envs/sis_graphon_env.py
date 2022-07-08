from .wrapper import GraphonEnv
from gym import spaces
import numpy as np


class SISGraphon(GraphonEnv):
    def __init__(self, env_config={}):
        super(SISGraphon,self).__init__()
        
        self.time_steps = env_config.get("time_steps", 50)
        self.adj_matrix = env_config.get("adj_matrix", np.array([[0.9,0.4],[0.4,0.9]]))
        
        # hyperparameters for state transition
        self.beta1 = env_config.get("beta1", 0.8)
        self.beta2 = env_config.get("beta2", 0)
        self.delta = env_config.get("delta", 0.3)
        
        # hyperparameters for reward function
        self.c1 = env_config.get("c1", 2)
        self.c2 = env_config.get("c2", 0.3)
        self.c3 = env_config.get("c3", 0.5)
        
        self.M = self.adj_matrix.shape[0]
        self.S = 2    # S, I
        self.A = 2    # C, NC
         
        self.state_space = spaces.Box(low=0,high=1,shape=(self.M,self.S))
        self.action_space = spaces.Box(low=-1,high=1,shape=(self.M*self.S*self.A,))
        
        if self.time_obs_augment:
            self.observation_space = spaces.Tuple([self.state_space, spaces.Box(0,self.time_steps,shape=())])
        else:
            self.observation_space = self.state_space
            
        self.t = None
        self.mu = None
        self.reset()
        

    
    
    def get_reward(self,mu,mu_g,pi):
        r = 0
        for m in range(self.M):
            r += mu[m][1] * self.c1
            r += mu[m][1] * pi[m][1][0] * self.c3
            for s in range(self.S):
                r += mu[m][s] * pi[m][s][1] * self.c2
        
        r = r/self.M
        
        r *= -1
        
        return r
        
    
    def aggregate_transition(self,mu,mu_g,pi):
        next_state = np.zeros((self.M,self.S))
        for m in range(self.M):
            next_state[m][0] += self.delta * mu[m][1]
            next_state[m][0] += (1 - self.beta1 * mu_g[m][1] * pi[m][0][0] - \
                self.beta2 * mu_g[m][1] * pi[m][0][1]) * mu[m][0]
            
            next_state[m][1] += self.beta1 * mu_g[m][1] * mu[m][0] * pi[m][0][0]
            next_state[m][1] += self.beta2 * mu_g[m][1] * mu[m][0] * pi[m][0][1]
            next_state[m][1] += (1-self.delta) * mu[m][1]
            
        
        return next_state
            
            

    
    
    

def test_env():
    env = SISGraphon()
    episode_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        #action = np.array([0, 1, 0, 1])

        obs, reward, done, info = env.step(action)

        episode_reward += reward
        if done:
            env.reset()
        
    
    print("episode_reward: ",episode_reward)



        
    
    
    

    


if __name__ == '__main__':
    test_env()

 


    
    