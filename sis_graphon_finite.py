import argparse
from ray.tune.registry import register_env
from ray.rllib.agents import ppo, ddpg, qmix
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.agents.qmix import QMixTrainer
from ray.rllib.examples.models.shared_weights_model import TorchSharedWeightsModel
from ray import tune
from gym.spaces import Tuple, Discrete, MultiDiscrete, Dict

from envs import SISGraphonNPlayer
from graphon import *


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="qmix")
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--graphon-type", type=int, default=2)
    parser.add_argument("--graphon-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    if args.graphon_type == 1:
        args.adj_matrix = erdos_renyi(args.graphon_size)
    elif args.graphon_type == 2:
        args.adj_matrix = stochastic_block(args.graphon_size)
    elif args.graphon_type == 3:
        args.adj_matrix = random_geometric(args.graphon_size)
        
    return args

def train_ippo(args):
    register_env("sis_graphon_finite-v0",lambda config: SISGraphonNPlayer(config))
    
    env_config={
        "num_players": args.num_agents,  
        "adj_matrix": args.adj_matrix,
    }
    env = SISGraphonNPlayer(env_config)
    
    obs_space = env.observation_space
    act_space = env.action_space
    
    ModelCatalog.register_custom_model("shared_weights_model", TorchSharedWeightsModel)
    
    
    config = ppo.DEFAULT_CONFIG.copy()
    
    
    
    config.update({
        "env": "sis_graphon_finite-v0",
        "env_config": env_config,
        #"num_gpus": 1,
        #"num_envs_per_worker": 5,
        "num_workers": 7,
        "framework": "torch",
        "train_batch_size": 256,
        #"lr":1e-3,
        "lr_schedule": [[0, 1e-3], [2000000, 1e-8]],
        "rollout_fragment_length": 16,
        "gamma": 0.95,
        "seed": 0,
        "model": {
            "custom_model": 'shared_weights_model',
            "custom_model_config":{},
        },
        "multiagent":{
            "policies":{
                "shared_policy": (PPOTorchPolicy, obs_space, act_space, {})
            },
            "policy_mapping_fn": (
                lambda agent_id, episode, **kwargs: "shared_policy"
            ),
        },
    })
    
    
    tune.run("PPO",
             config=config,
             local_dir="results/sis_graphon_finite_{}_{}_{}".format(
args.num_agents, args.graphon_type,args.adj_matrix.shape[0]),
             stop={"training_iteration":200},
             checkpoint_freq = 50,
             checkpoint_at_end = True,
    )

def train_iql(args):
    register_env("sis_graphon_finite-v0",lambda config: SISGraphonNPlayer(config))
    
    env_config={
        "num_players": args.num_agents,  
        "adj_matrix": args.adj_matrix,
    }
    env = SISGraphonNPlayer(env_config)
    
    obs_space = env.observation_space
    act_space = env.action_space
    
    ModelCatalog.register_custom_model("shared_weights_model", TorchSharedWeightsModel)
    
    tune.run(
        "DQN",
        stop={"training_iteration":100},
        local_dir="results/sis_graphon_finite_{}_{}_{}".format(
args.num_agents, args.graphon_type, args.adj_matrix.shape[0]),
        checkpoint_freq=20,
        config={
            # Enviroment specific.
            "env": "sis_graphon_finite-v0",
            "env_config": env_config,
            "framework": "torch",
            # General
            #"num_gpus": 1,
            "num_workers": 7,
            "gamma": 0.95,
            "lr_schedule": [
                [0, 5e-4],
                [20000000, 1e-8],
            ],
            "double_q": True,
            "dueling": True,
            "num_atoms": 1,
            "noisy": False,
            "prioritized_replay": False,
            "n_step": 1,
            "hiddens": [64],
            "target_network_update_freq": 8000,
            "adam_epsilon": 0.00015,
            "learning_starts": 20000,
            "buffer_size": 8000,
            "rollout_fragment_length": 32,
            "train_batch_size": 128,
            # Method specific.
            "model": {
                "custom_model": 'shared_weights_model',
                "custom_model_config":{},
            },
            "multiagent":{
                "policies":{
                    "shared_policy": (DQNTorchPolicy, obs_space, act_space, {})
                },
                "policy_mapping_fn": (
                    lambda agent_id, episode, **kwargs: "shared_policy"
                ),
            },
            
            
        },
    )

def train_vdn(args): 
    env_config={
        "num_players": args.num_agents,  
        "adj_matrix": args.adj_matrix,
    }
    env = SISGraphonNPlayer(env_config)
    obs_space = env.observation_space
    act_space = env.action_space
    
    grouping = {
        "group_1": list(range(env.N))
    }
    
    obs_space = Tuple([obs_space for _ in range(env.N)])
    act_space = Tuple([act_space for _ in range(env.N)])
    
    register_env(
            "sis_graphon_finite-v1",
            lambda config: SISGraphonNPlayer(config).with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space
            ),
    )
    
    config = qmix.DEFAULT_CONFIG
    
    config.update({
        "env": "sis_graphon_finite-v1",
        "env_config": env_config,
        "framework":"torch",
        "mixer": "vdn",
        "gamma": 0.95,
        "num_envs_per_worker": 5, 
        "num_workers": 7,
        "lr": 5e-5,
        "mixing_embed_dim": 32,
        "model": {
            "lstm_cell_size": 32,
            "max_seq_len": 999999,
        },
        "buffer_size": 2000,
        "target_network_update_freq": 4000,
        "rollout_fragment_length": 32,
        "train_batch_size": 128,
    })
    
    
    
    tune.run("QMIX",
             config=config,
             local_dir="results/sis_graphon_finite_{}_{}_{}".format(
args.num_agents, args.graphon_type, args.adj_matrix.shape[0]),
             stop={"training_iteration":100},
             checkpoint_freq = 10,
             checkpoint_at_end = True,
    )
    
    

def train_qmix(args): 
    env_config={
        "num_players": args.num_agents,  
        "adj_matrix": args.adj_matrix,
    }
    env = SISGraphonNPlayer(env_config)
    obs_space = env.observation_space
    act_space = env.action_space
    
    grouping = {
        "group_1": list(range(env.N))
    }
    
    obs_space = Tuple([obs_space for _ in range(env.N)])
    act_space = Tuple([act_space for _ in range(env.N)])
    
    register_env(
            "sis_graphon_finite-v1",
            lambda config: SISGraphonNPlayer(config).with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space
            ),
    )
    
    config = qmix.DEFAULT_CONFIG
    
    config.update({
        "env": "sis_graphon_finite-v1",
        "env_config": env_config,
        "framework":"torch",
        "mixer": "qmix",
        "gamma": 0.95,
        "num_envs_per_worker": 5, 
        "num_workers": 7,
        "lr": 5e-5,
        "mixing_embed_dim": 64,
        "model": {
            "use_lstm": False,
            "lstm_cell_size": 32,
            "max_seq_len": 999999,
        },
        "buffer_size": 2000,
        "target_network_update_freq": 5000,
        "rollout_fragment_length": 32,
        "train_batch_size": 128,
    })
    
    
    
    tune.run("QMIX",
             config=config,
             local_dir="results/sis_graphon_finite_{}_{}_{}".format(
args.num_agents, args.graphon_type, args.adj_matrix.shape[0]),
             stop={"training_iteration":100},
             checkpoint_freq = 10,
             checkpoint_at_end = True,
    )

def test_qmix(args):
    env_config={
        "num_players": args.num_agents,  
        "adj_matrix": args.adj_matrix,
    }
    env = SISGraphonNPlayer(env_config)
    obs_space = env.observation_space
    act_space = env.action_space
    
    grouping = {
        "group_1": list(range(env.N))
    }
    
    obs_space = Tuple([obs_space for _ in range(env.N)])
    act_space = Tuple([act_space for _ in range(env.N)])
    
    register_env(
            "sis_graphon_finite-v1",
            lambda config: SISGraphonNPlayer(config).with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space
            ),
    )
    
    config = qmix.DEFAULT_CONFIG
    
    config.update({
        "env": "sis_graphon_finite-v1",
        "env_config": env_config,
        "framework":"torch",
        "mixer": "qmix",
        "gamma": 0.95,
        "num_envs_per_worker": 5, 
        "num_workers": 7,
        "lr": 5e-5,
        "mixing_embed_dim": 32,
        "model": {
            "lstm_cell_size": 32,
            "max_seq_len": 999999,
        },
        "buffer_size": 2000,
        "target_network_update_freq": 500,
        "rollout_fragment_length": 32,
        "train_batch_size": 64,
    })
    
    agent = qmix.QMixTrainer(env='sis_graphon_finite-v1',config=config)
    
    agent.restore("results/sis_graphon_finite_20_1_10/QMIX/QMIX_sis_graphon_finite-v1_95d22_00000_0_2022-05-23_14-50-08/checkpoint_000100/checkpoint-100")
    
    init_prev_a = prev_a = None
    init_prev_r = prev_r = None
    
    lstm_cell_size = config["model"]["lstm_cell_size"]
    init_state = state = [np.zeros([env.N,lstm_cell_size], np.float32) for _ in range(2)]
    
    episode_reward = []
    eval_num = 1000
    np.random.seed(args.seed)
    for T in range(eval_num):
        obs = env.reset()
        total_reward = 0
        for step in range(50):
            u, state_out, _ = agent.compute_single_action(
                observation=env.x,
                state=state,
                prev_action=prev_a,
                prev_reward=prev_r,
                explore=False,
            )
    
            obs, reward, done, info = env.step(u)
            reward = [reward[idx] for idx in reward.keys()]
            
            
            total_reward += sum(reward)
            
            state = state_out
            if init_prev_a is not None:
                prev_a = a
            if init_prev_r is not None:
                prev_r = reward
            
        
        print(total_reward)
        
        episode_reward.append(total_reward)
                
    
    print("mean episode reward: ", sum(episode_reward)/eval_num)


if __name__ == '__main__':
    args = arg_parse()
    
    
    if args.algo == 'ippo':
        train_ippo(args)
    elif args.algo == 'iql':
        train_iql(args)
    elif args.algo == 'qmix':
        train_qmix(args)
        #test_qmix(args)
    elif args.algo == 'vdn':
        train_vdn(args)