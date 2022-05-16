import gym
from gym import spaces
import numpy as np


class SISGraphon(gym.Env):
    def __init__(self, env_config={}):
        super(SISGraphon,self).__init__()
        
        
        self.time_obs_augment = env_config.get("time_obs_augment", False)
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
    
    
    

def test_env():
    env = SISGraphon()

    for step in range(100):
        action = env.action_space.sample()
        #action = np.array([0, 1, 0, 1])

        obs, reward, done, info = env.step(action)

        print(reward)
        if done:
            env.reset()



        
    

if __name__ == '__main__':
    test_env()

 


    
    