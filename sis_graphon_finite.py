from ray.tune.registry import register_env
from ray.rllib.agents import ppo, ddpg
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.examples.models.shared_weights_model import TorchSharedWeightsModel
from ray import tune

from envs import SISGraphonNPlayer



def env_creator(env_config=None):
    return SISGraphonNPlayer()


def train_ppo():
    register_env("sis_graphon_finite-v0",env_creator)
    
    env = SISGraphonNPlayer()
    obs_space = env.observation_space
    act_space = env.action_space
    
    ModelCatalog.register_custom_model("shared_weights_model", TorchSharedWeightsModel)
    
    
    config = ppo.DEFAULT_CONFIG.copy()
    
    
    
    config.update({
        "env": "sis_graphon_finite-v0",
        #"num_gpus": 1,
        #"num_workers": 8,
        "framework": "torch",
        "train_batch_size": 1024,
        #"lr":1e-3,
        "lr_schedule": [[0, 1e-4], [1000000, 1e-8]],
        "rollout_fragment_length": 64,
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
             local_dir="sis_graphon_finite-v0",
             stop={"training_iteration":100},
             checkpoint_freq = 50,
             checkpoint_at_end = True,
    )

def train_iqn():
    register_env("sis_graphon_finite-v0",env_creator)
    
    env = SISGraphonNPlayer()
    obs_space = env.observation_space
    act_space = env.action_space
    
    ModelCatalog.register_custom_model("shared_weights_model", TorchSharedWeightsModel)
    
    tune.run(
        "DQN",
        stop={"training_iteration":100},
        local_dir="sis_graphon_finite-v0",
        checkpoint_freq=20,
        config={
            # Enviroment specific.
            "env": "sis_graphon_finite-v0",
            "framework": "torch",
            # General
            #"num_gpus": 1,
            "num_workers": 7,
            "gamma": 0.95,
            "lr_schedule": [
                [0, 1e-5],
                [20000000, 1e-8],
            ],
            "double_q": True,
            "dueling": True,
            "num_atoms": 1,
            "noisy": False,
            "prioritized_replay": False,
            "n_step": 1,
            "target_network_update_freq": 8000,
            "adam_epsilon": 0.00015,
            "learning_starts": 20000,
            "buffer_size": int(1e6),
            "rollout_fragment_length": 64,
            "train_batch_size": 512,
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

    

if __name__ == '__main__':
    #train_ppo()
    train_iqn()