import argparse
import gym
from gym import spaces
import numpy as np

from ray.tune.registry import register_env
from ray.rllib.agents import ddpg, ppo
from ray.rllib.models import ModelCatalog

from models import GraphonModel
from envs import SISGraphon, SISGraphonNPlayer
from graphon import *

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--graphon-type", type=int, default=2)
    parser.add_argument("--graphon-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    if args.graphon_type == 1:
        args.adj_matrix = erdos_renyi(args.graphon_size)
        args.model_path = "results/sis_graphon_1_10/PPO/best/checkpoint_001000/checkpoint-1000"
        
    elif args.graphon_type == 2:
        args.adj_matrix = stochastic_block(args.graphon_size)
        args.model_path = "results/sis_graphon_2_10/PPO/best/checkpoint_001000/checkpoint-1000"
        #args.model_path = "results/sis_graphon_2_20/PPO/best/checkpoint_001000/checkpoint-1000"
        
    elif args.graphon_type == 3:
        args.adj_matrix = random_geometric(args.graphon_size)
        args.model_path = "results/sis_graphon_3_10/PPO/best/checkpoint_001000/checkpoint-1000"
    
    
    return args
    
def main(args):
    register_env("sis_graphon-v0",lambda config: SISGraphon(config))

    
    model_name = 'graphon_model'
    ModelCatalog.register_custom_model(model_name, GraphonModel)
    
    
    config = ppo.DEFAULT_CONFIG.copy()
    config.update({
        "env": "sis_graphon-v0",
        "env_config":{
            "adj_matrix": args.adj_matrix,
        },
        "framework": "torch",
        "model": {
            "custom_model": 'graphon_model',
            "custom_model_config":{},
        },
    })
    
    
    
    
    agent = ppo.PPOTrainer(env='sis_graphon-v0',config=config)
    
    agent.restore(args.model_path)
    
    env_g = SISGraphon({"adj_matrix": args.adj_matrix})
    
    env_config = {
        "num_players":args.num_agents,
        "adj_matrix": args.adj_matrix,
    }
    env = SISGraphonNPlayer(env_config)
    episode_reward = []
    eval_num = 1000
    np.random.seed(args.seed)
    for T in range(eval_num):
        obs = env.reset()
        total_reward = 0
        for step in range(50):
            mu = env.dist_g()                                       #compute state distribution
            pi = agent.compute_single_action(observation=mu)        #sample from policy ensemble
            
            pi = env_g.act_transform(pi) 
            action = []
            
            for agent_id in range(env.N):
                agent_state = env.x[agent_id]
                prob = pi[agent_state[0]][agent_state[1]]
                agent_act = np.random.choice(range(env.A),p=prob)
                action.append(agent_act)
    
            obs, reward, done, info = env.step(action)
            reward = [reward[idx] for idx in reward.keys()]
            
            
            total_reward += sum(reward)
        
        print(total_reward)
        
        episode_reward.append(total_reward)
                
    reward_std = np.std(episode_reward)
    print("mean episode reward: ", sum(episode_reward)/eval_num)
    print("std episode reward: ", reward_std)


if __name__ == '__main__':
    args = arg_parse()
    main(args)