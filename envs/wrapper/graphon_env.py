import gym
from gym import spaces
import numpy as np


class GraphonEnv(gym.Env):
    def __init__(self, env_config={}):
        super(GraphonEnv,self).__init__()
        
        
        self.time_obs_augment = env_config.get("time_obs_augment", False)
        self.time_steps = env_config.get("time_steps", 50)
        self.adj_matrix = env_config.get("adj_matrix", np.array([[0.9,0.4],[0.4,0.9]]))

        
        
        self.M = self.adj_matrix.shape[0]
        self.S = env_config.get("num_states", 2)
        self.A = env_config.get("num_actions", 2)
         
        self.state_space = spaces.Box(low=0,high=1,shape=(self.M,self.S))
        self.action_space = spaces.Box(low=-1,high=1,shape=(self.M*self.S*self.A,))
        
        if self.time_obs_augment:
            self.observation_space = spaces.Tuple([self.state_space, spaces.Box(0,self.time_steps,shape=())])
        else:
            self.observation_space = self.state_space
            
        self.t = None
        self.mu = None
        
    
    def reset(self):
        self.t = 0
        self.mu = self.state_space.sample()
            
        for m in range(self.M):
            self.mu[m][:] = np.exp(self.mu[m][:])/sum(np.exp(self.mu[m][:]))
            
        #self.mu = np.array([[0.5,0.5],[0.5,0.5]])
        obs = self.obs_transform(self.t,self.mu)
        
        return obs
    
    def graphon_mean_field(self,mu):
        mu_g = np.zeros((self.M,self.S))
        for i in range(self.M):
            for s in range(self.S):
                mu_g[i][s] = sum([mu[j][s]*self.adj_matrix[i][j] \
                                  for j in range(self.M)]) / self.M
        return mu_g
    
    
    def get_reward(self,mu,mu_g,pi):
        pass
        
    
    def aggregate_transition(self,mu,mu_g,pi):
        pass
            
            
    
    def act_transform(self,act):
        #act = np.concatenate(act)
        #pi = act.reshape((self.M,self.S,self.A-1))
        #pi_new = np.zeros((self.M,self.S,self.A))                                    
        #                                    
        #for m in range(self.M):
        #    pi_new[m][0][0] = pi[m][0][0]
        #    pi_new[m][0][1] = 1-pi[m][0][0]
        #    pi_new[m][1][0] = pi[m][1][0]
        #    pi_new[m][1][1] = 1 - pi[m][1][0]
         

        pi = act.reshape((self.M,self.S,self.A))
        pi_new = pi.copy()
        
        for m in range(self.M):
            for s in range(self.S):                        
                pi_new[m][s][:] = np.exp(pi[m][s][:])/sum(np.exp(pi[m][s][:]))
    
        return pi_new
    
    
    def obs_transform(self,t,obs):
        if self.time_obs_augment:
            return tuple([obs,np.array(t,dtype=np.float32)])
        else:
            return obs
            #return obs.reshape((self.M*self.S,))
    
    def seed(self,seed):
        np.random.seed(seed)
    
    def render(self):
        pass
    
    def close(self):
        pass
        
    def step(self,act):
        pi = self.act_transform(act)
        mu_g = self.graphon_mean_field(self.mu)
        
        next_state = self.aggregate_transition(self.mu,mu_g,pi)
        observation = self.obs_transform(self.t+1,next_state)
        reward = self.get_reward(self.mu,mu_g,pi)
        done = self.t >= self.time_steps
        
        self.mu = next_state
        self.t += 1
        
        return observation, reward, done, {}
    
    

 


    
    