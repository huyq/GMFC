import sys
import gym
from gym.spaces import Tuple, Discrete, MultiDiscrete
from ray.rllib import MultiAgentEnv
import numpy as np


class SISGraphonNPlayer(MultiAgentEnv):
    def __init__(self,time_obs_augment=False):
        super(SISGraphonNPlayer,self).__init__()
        
        self.N = 20
        
        self.time_obs_augment = time_obs_augment
        self.time_steps = 50
        self.adj_matrix = np.array([[0.9,0.4],[0.4,0.9]])

        
        # hyperparameters for state transition
        self.beta1 = 0.8
        self.beta2 = 0
        self.delta = 0.3
        
        # hyperparameters for reward function
        self.gamma = 0.95
        self.c1 = 2
        self.c2 = 0.3
        self.c3 = 0.5
        
        self.M = self.adj_matrix.shape[0]
        self.S = 2    # S, I
        self.A = 2    # C, NC
         
        self.state_space = MultiDiscrete([self.M,self.S])
        self.action_space = Discrete(self.A)
        self._agent_ids = set(range(self.N))

        
        if self.time_obs_augment:
            self.observation_space = spaces.Tuple([self.state_space, spaces.Box(0,self.time_steps,shape=())])
        else:
            self.observation_space = self.state_space
            
        self.t = None
        self.x = None
        self.reset()
        
    
    def reset(self):
        self.t = 0
        self.x = [self.state_space.sample() for _ in range(self.N)]
            
        #for i in range(self.N):
        #    for j in range(i+1,self.N):
        #        edge_prob = self.graphon[self.x[i][0],self.x[j][0]]
        #        edge = np.random.choice([0, 1], p=[1-edge_prob, edge_prob])
        #        self.adj_matrix[i][j] = edge
        #        self.adj_matrix[j][i] = edge
            

        obs = {agent_id: self.x[agent_id] for agent_id in range(self.N)}
        
        return obs
    
    def dist_g(self):
        mu = np.zeros((self.M,self.S))
        m_ = np.zeros(2)
        for _x in self.x:
            m = _x[0]
            s = _x[1]
            mu[m][s] += 1
            m_[m] += 1
        
        for m in range(self.M):
            mu[m][:] /= m_[m]
        
        return mu
    
    
    def get_total_reward(self,x,u,g):
        r = 0
        mu = np.zeros(self.S)
        pi = np.zeros((self.S,self.A))
        for i in range(self.N):
            s = x[i]
            a = u[i]
            mu[s] += 1
            pi[s][a] += 1
            
        r += mu[1] * self.c1
        r += pi[1][0] * self.c3
        r += (pi[0][1] + pi[1][1]) * self.c2
        
        r /= self.N
        
        r *= -1
        return r
    
    def get_reward(self,obs,a,g):
        s = obs[1]
        r = - s * self.c1 - s * (1-a) *self.c3 - a * self.c2
        return r / self.N
    
    def graphon_mean_field(self, x, node_index):
        mu_g = np.zeros(self.S)
        alpha = x[node_index][0]
        for _x in x:
            beta = _x[0]
            mu_g[_x[1]] += self.adj_matrix[alpha][beta]
        
        mu_g /= self.N
            
        return mu_g
        
        
    
    def transition_probs_g(self,t, x, u, g):
        if x[1] == 0:
            if u == 0:
                transition_prob = self.beta1 * g[1]
            if u == 1:
                transition_prob = self.beta2 * g[1]
            return np.array([1 - transition_prob, transition_prob])
        
        elif x[1] == 1:
            transition_prob = self.delta
            return np.array([transition_prob, 1 - transition_prob])

    
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
        for node_index in range(self.N):
            G = self.graphon_mean_field(self.x, node_index)
            transition_prob = self.transition_probs_g(self.t, self.x[node_index], u[node_index], G)
            next_state.append(tuple([self.x[node_index][0], np.random.choice(range(self.S), 1, None,
                p=transition_prob).item()]))
            
            
            reward[node_index] = self.get_reward(self.x[node_index],u[node_index],G)
            
        self.x = next_state
        self.t += 1
        
        observation = {node_idx:next_state[node_idx] for node_idx in range(len(next_state))}
        done = self.t >= self.time_steps
        done = {idx:done for idx in range(self.N)}
        done['__all__'] = self.t >= self.time_steps
        
        return observation, reward, done, {}



def test_env():
    env = SISGraphonNPlayer()

    total_reward = 0
    for step in range(50):
        action = [env.action_space.sample() for _ in range(env.N)]
        #action = np.array([0, 1, 0, 1])

        obs, reward, done, info = env.step(action)
        reward = [reward[idx] for idx in reward.keys()]
        
        total_reward += sum(reward)/env.N
        if done:
            env.reset()
    
    print("episode reward: ", total_reward)
    

if __name__ == '__main__':
    test_env()
