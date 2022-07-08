import sys
from .wrapper import GraphonNPlayerEnv
from gym.spaces import Tuple, Discrete, MultiDiscrete, Dict, Box
from ray.rllib import MultiAgentEnv
import numpy as np


class SISGraphonNPlayer(GraphonNPlayerEnv):
    def __init__(self, env_config={}):
        super(SISGraphonNPlayer,self).__init__()
        
        self.N = env_config.get("num_players", 20)
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
        
         
        self.state_space = MultiDiscrete([self.M,self.S])
        self.action_space = Discrete(self.A)
        self._agent_ids = set(range(self.N))

        
        self.observation_space = self.state_space
            
        self.t = None
        self.x = None
        self.reset()
        
    
    
    def get_reward(self,obs,a,g):
        s = obs[1]
        r = - s * self.c1 - s * (1-a) *self.c3 - a * self.c2
        return r / self.N
        
        
    
    def transition_probs_g(self, obs, a, g):
        if obs[1] == 0:
            if a == 0:
                transition_prob = self.beta1 * g[1]
            if a == 1:
                transition_prob = self.beta2 * g[1]
            return np.array([1 - transition_prob, transition_prob])
        
        elif obs[1] == 1:
            transition_prob = self.delta
            return np.array([transition_prob, 1 - transition_prob])
    
    def state_transition(self, obs, a, g):
        transition_prob = self.transition_probs_g(obs, a, g)
        new_s = np.random.choice(range(self.S), 1, None, p=transition_prob).item()
        return tuple([obs[0],new_s])
        




def test_env():
    import sys
    sys.path.append('..')
    from graphon import stochastic_block, random_geometric
    
    
    env_config={
        "num_players": 20,  
        "adj_matrix": stochastic_block(10),
    }
    env = SISGraphonNPlayer(env_config)
    env.reset()

    total_reward = 0
    for step in range(50):
        action = [env.action_space.sample() for _ in range(env.N)]
        #action = np.array([0, 1, 0, 1])
        
        obs, reward, done, info = env.step(action)
        reward = [reward[idx] for idx in reward.keys()]

        
        total_reward += sum(reward)
        if done:
            env.reset()
    
    print("episode reward: ", total_reward)
    

if __name__ == '__main__':
    test_env()
