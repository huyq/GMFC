import argparse
import gym
from gym import spaces
import numpy as np

from ray.tune.registry import register_env
from ray.rllib.agents import ddpg, ppo
from ray.rllib.models import ModelCatalog

from models import GraphonModel
from envs import SISGraphon, SISGraphonNPlayer

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", type=int, default=20)
    
    args = parser.parse_args()
    
    return args
    
def main(args):
    def env_creator(env_config=None):
        return SISGraphon()
    
    register_env("sis_graphon-v0",env_creator)
    
    model_name = 'sis_graphon'
    ModelCatalog.register_custom_model(model_name, GraphonModel)
    
    
    config = ppo.DEFAULT_CONFIG.copy()
    config.update({
        "framework": "torch",
        "train_batch_size": 256,
        #"lr":1e-3,
        "lr_schedule": [[0, 1e-3], [1000000, 1e-7]],
        "rollout_fragment_length": 8,
        "gamma": 0.95,
        "seed": 0,
        "model": {
            "custom_model": 'sis_graphon',
            "custom_model_config":{},
        },
    })
    
    
    agent = ppo.PPOTrainer(env='sis_graphon-v0',config=config)
    
    agent.restore("sis_graphon-v0/PPO/PPO_sis_graphon-v0_ba8bc_00000_0_2022-05-16_15-28-06/checkpoint_000200/checkpoint-200")
    
    env_g = SISGraphon()
    
    env_config = {"num_players":args.num_agents}
    env = SISGraphonNPlayer(env_config)
    episode_reward = []
    eval_num = 10
    for T in range(eval_num):
        obs = env.reset()
        total_reward = 0
        for step in range(50):
            mu = env.dist_g()
            pi = agent.compute_single_action(observation=mu)
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
                
    
    print("mean episode reward: ", sum(episode_reward)/eval_num)


if __name__ == '__main__':
    args = arg_parse()
    main(args)