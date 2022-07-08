import argparse
from ray.tune.registry import register_env
from ray.rllib.agents import ppo, ddpg
from ray.rllib.models import ModelCatalog
from ray import tune

from envs import SISGraphon
from models import GraphonModel
from graphon import *


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--graphon-type", type=int, default=2)
    parser.add_argument("--graphon-size", type=int, default=2)
    
    args = parser.parse_args()
    if args.graphon_type == 1:
        args.adj_matrix = erdos_renyi(args.graphon_size)
    elif args.graphon_type == 2:
        args.adj_matrix = stochastic_block(args.graphon_size)
    elif args.graphon_type == 3:
        args.adj_matrix = random_geometric(args.graphon_size)
        
    return args

    
def test_single_step(agent):
    env = SISGraphon()
    mu = [np.array([[0.9,0.1],[0.9,0.1]]),np.array([[0.5,0.5],[0.5,0.5]]),np.array([[0.1,0.9],[0.1,0.9]]),np.array([[0.1,0.9],[0.9,0.1]])]
    for obs in mu:
        obs = env.obs_transform(t=0,obs=obs)    
        action = agent.compute_single_action(observation=obs)
        pi = env.act_transform(action)
        print(pi)
    
    
    
def train_ppo(args):
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
        #"num_workers": 8,
        "train_batch_size": 128,
        #"lr":1e-3,
        "lr_schedule": [[0, 5e-4], [20000000, 1e-8]],
        "rollout_fragment_length": 10,
        #"entropy_coeff": 0.001,
        "gamma": 0.95,
        "seed": 0,
        "model": {
            "custom_model": 'graphon_model',
            "custom_model_config":{},
        },
    })
    
    
    tune.run("PPO",
             config=config,
             local_dir="results/sis_graphon_{}_{}".format(args.graphon_type,args.graphon_size),
             stop={"training_iteration":1000},
             checkpoint_freq = 10,
             checkpoint_at_end = True,
    )

    
    
def train_ddpg(args):    
    register_env("sis_graphon-v0",lambda config: SISGraphon(config))

    
    config = ddpg.ddpg.DEFAULT_CONFIG.copy()
    config["num_workers"] = 8
    config["actor_hiddens"] = [400,300]
    config["critic_hiddens"] = [400,300]
    config["critic_lr"] = 1e-4
    config["actor_lr"] = 1e-5
    config["critic_hidden_activation"] = "tanh"
    config["actor_hidden_activation"] = "tanh"  
    config["replay_buffer_config"]["capacity"] = 100000
    config["train_batch_size"] = 256
    config["policy_delay"] = 1
    config["tau"] = 0.001
    config["rollout_fragment_length"] = 8
    config["gamma"] = 0.95
    config["env_config"] = {"adj_matrix": args.adj_matrix,}
    
    
    agent1 = ddpg.DDPGTrainer(env='sis_graphon-v0',config=config)

    
    for _ in range(100):
        result = agent1.train()
        print(_, result['episode_reward_mean'])
    

    test_single_step(agent1)
    
    agent1.stop()
    



def train_td3(args):
    register_env("sis_graphon-v0",lambda config: SISGraphon(config))

    
    config = ddpg.td3.TD3_DEFAULT_CONFIG.copy()
    config["num_workers"] = 8
    config["actor_hiddens"] = [512,512]
    config["critic_hiddens"] = [512,512]
    config["critic_lr"] = 1e-5
    config["actor_lr"] = 1e-6
    config["critic_hidden_activation"] = "tanh"
    config["actor_hidden_activation"] = "tanh"  
    config["replay_buffer_config"]["capacity"] = 100000
    config["train_batch_size"] = 256
    config["policy_delay"] = 1
    config["tau"] = 0.001
    config["rollout_fragment_length"] = 16
    config["gamma"] = 0.95
    config["env_config"] = {"adj_matrix": args.adj_matrix,}
    
    agent1 = ddpg.TD3Trainer(env='sis_graphon-v0',config=config)
    
    

    for iter in range(50):
        result = agent1.train()
        print(iter, result['episode_reward_mean'])
    
    
    test_single_step(agent1)
   
    agent1.stop()
    


if __name__ == '__main__':
    args = arg_parse()
    
    if args.algo == 'ppo':
        train_ppo(args)
    elif args.algo == 'ddpg':
        train_ddpg(args)
    elif args.algo == 'td3':
        train_td3(args)