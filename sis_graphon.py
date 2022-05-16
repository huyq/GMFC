from ray.tune.registry import register_env
from ray.rllib.agents import ppo, ddpg
from ray.rllib.models import ModelCatalog
from ray import tune

from envs import SISGraphon
from models import GraphonModel


def env_creator(env_config=None):
    return SISGraphon()
    
def test_single_step(agent):
    env = SISGraphon()
    mu = [np.array([[0.9,0.1],[0.9,0.1]]),np.array([[0.5,0.5],[0.5,0.5]]),np.array([[0.1,0.9],[0.1,0.9]]),np.array([[0.1,0.9],[0.9,0.1]])]
    for obs in mu:
        obs = env.obs_transform(t=0,obs=obs)    
        action = agent.compute_single_action(observation=obs)
        pi = env.act_transform(action)
        print(pi)
    
    
    
def train_ppo():
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
        "env": "sis_graphon-v0",
        "model": {
            "custom_model": 'sis_graphon',
            "custom_model_config":{},
        },
    })
    
    
    tune.run("PPO",
             config=config,
             local_dir="sis_graphon-v0",
             stop={"training_iteration":100},
             checkpoint_freq = 10,
             checkpoint_at_end = True,
    )

    
    
def train_ddpg():    
    register_env("sis_graphon-v0",env_creator)

    
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
    
    
    agent1 = ddpg.DDPGTrainer(env='sis_graphon-v0',config=config)

    
    for _ in range(100):
        result = agent1.train()
        print(_, result['episode_reward_mean'])
    
    #state = agent1.save()
    
    #agent1.stop()
    
    #config["critic_lr"] = 5e-6
    #config["actor_lr"] = 1e-6
    #
    #agent2 = ddpg.DDPGTrainer(env='sis_graphon-v0',config=config)
    #agent2.restore(state)
    #
    #for _ in range(50):
    #    result = agent2.train()
    #    print(_, result['episode_reward_mean'])
    #
    #
    #agent2.stop()
    #
    #return
           
    
    mu = np.array([[0.5,0.5],[0.5,0.5]])
    #obs = tuple([mu,np.array(0, dtype=np.float32)])
    
    action = agent1.compute_single_action(observation=mu)
    env = SISGraphon()
    pi = env.act_transform(action)
    print(pi)
    agent1.stop()
    



def train_td3():
    register_env("sis_graphon-v0",env_creator)

    
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
    config["gamma"] = 0
    
    agent1 = ddpg.TD3Trainer(env='sis_graphon-v0',config=config)
    
    
    #f = open('output.txt','w+')
    for iter in range(50):
        result = agent1.train()
        #f.write(str(result['episode_reward_mean'])+'\n')
        print(iter, result['episode_reward_mean'])
    
    #f.close()
    
    
    mu = np.array([[0.5,0.5],[0.5,0.5]])
    #obs = tuple([mu,np.array(0, dtype=np.float32)])
    
    action = agent1.compute_single_action(observation=mu)
    env = SISGraphon()
    pi = env.act_transform(action)
    print(pi)
    
    agent1.stop()


if __name__ == '__main__':
    train_ppo()