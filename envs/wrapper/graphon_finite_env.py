import sys
import gym
from gym.spaces import Tuple, Discrete, MultiDiscrete, Dict, Box
from ray.rllib import MultiAgentEnv
import numpy as np


class GraphonNPlayerEnv(MultiAgentEnv):
    def __init__(self, env_config={}):
        super(GraphonNPlayerEnv,self).__init__()
        
        self.N = env_config.get("num_players", 20)
        self.time_steps = env_config.get("time_steps", 50)
        self.adj_matrix = env_config.get("adj_matrix", np.array([[0.9,0.4],[0.4,0.9]]))
        
        
        self.M = self.adj_matrix.shape[0]
        self.S = env_config.get("num_states", 2)
        self.A = env_config.get("num_actions", 2)
        
         
        self.state_space = MultiDiscrete([self.M,self.S])
        self.action_space = Discrete(self.A)
        self._agent_ids = set(range(self.N))

        
        self.observation_space = self.state_space
            
        self.t = None
        self.x = None
        
    
    def reset(self):
        self.t = 0
        self.x = [self.state_space.sample() for _ in range(self.N)]
        
        
        
        for i in range(self.N):
            m = i/self.N*self.M
            
            self.x[i][0] = int(m)
            
            #prob = np.array([0.7,0.3])
            #self.x[i][1] = np.random.choice(range(self.S), 1, None, p=prob).item()
                        
        

        obs = {agent_id: self.x[agent_id] for agent_id in range(self.N)}
        
        return obs
    
    def dist_g(self):
        mu = np.zeros((self.M,self.S))
        m_ = np.zeros(self.M)
        for _x in self.x:
            m = _x[0]
            s = _x[1]
            mu[m][s] += 1
            m_[m] += 1
        
        for m in range(self.M):
            if m_[m] > 0:
                mu[m][:] /= m_[m]
        
        return mu
    
    
    
    def graphon_mean_field(self, x, agent_id):
        mu_g = np.zeros(self.S)
        alpha = x[agent_id][0]
        for _x in x:
            beta = _x[0]
            mu_g[_x[1]] += self.adj_matrix[alpha][beta]
        
        mu_g /= self.N
        
        return mu_g
    
    def get_reward(self,obs,a,g):
        pass
        
    
    def state_transition(self, obs, a, g):
        pass
        

    
    def obs_transform(self,t,obs):
        if self.time_obs_augment:
            return tuple([obs,np.array(t,dtype=np.float32)])
        else:
            return obs
    
    
    def seed(self,seed):
        np.random.seed(seed)
    
    def render(self):
        pass
    
    def close(self):
        pass
        
    def step(self,u):
        next_state = []
        reward = {}
        for agent_id in range(self.N):
            G = self.graphon_mean_field(self.x, agent_id)
            next_state.append(self.state_transition(self.x[agent_id],u[agent_id],G))
            reward[agent_id] = self.get_reward(self.x[agent_id],u[agent_id],G)
            
        self.x = next_state
        self.t += 1
        
     
        observation = {agent_id: next_state[agent_id] for agent_id in range(self.N)}
        done = {"__all__": self.t >= self.time_steps}
        
        return observation, reward, done, {}

