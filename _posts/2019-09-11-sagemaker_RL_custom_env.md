---
layout: posts
author: Huan
title: Reinforcement learning custom environment in Sagemaker with Ray (RLlib)
---

Demo setup for simple (reinforcement learning) custom environment in Sagemaker.
This example uses Proximal Policy Optimization with Ray (RLlib).

---

Code on my [Github](https://github.com/ChuaCheowHuan/sagemaker_Ray_RLlib_custom_env)

---

**The training script:**

```
import json
import os

import gym
import ray
from ray.tune import run_experiments
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from mod_op_env import ArrivalSim

from sagemaker_rl.ray_launcher import SageMakerRayLauncher

"""
def create_environment(env_config):
    import gym
#     from gym.spaces import Space
    from gym.envs.registration import register
    # This import must happen inside the method so that worker processes import this code
    register(
        id='ArrivalSim-v0',
        entry_point='env:ArrivalSim',
        kwargs= {'price' : 40}
    )
    return gym.make('ArrivalSim-v0')
"""
def create_environment(env_config):
    price = 30.0
    # This import must happen inside the method so that worker processes import this code
    from mod_op_env import ArrivalSim
    return ArrivalSim(price)


class MyLauncher(SageMakerRayLauncher):
    def __init__(self):        
        super(MyLauncher, self).__init__()
        self.num_gpus = int(os.environ.get("SM_NUM_GPUS", 0))
        self.hosts_info = json.loads(os.environ.get("SM_RESOURCE_CONFIG"))["hosts"]
        self.num_total_gpus = self.num_gpus * len(self.hosts_info)

    def register_env_creator(self):
        register_env("ArrivalSim-v0", create_environment)

    def get_experiment_config(self):
        return {
          "training": {
            "env": "ArrivalSim-v0",
            "run": "PPO",
            "stop": {
              "training_iteration": 3,
            },

            "local_dir": "/opt/ml/model/",
            "checkpoint_freq" : 3,

            "config": {                                
              #"num_workers": max(self.num_total_gpus-1, 1),
              "num_workers": max(self.num_cpus-1, 1),
              #"use_gpu_for_workers": False,
              "train_batch_size": 128, #5,
              "sample_batch_size": 32, #1,
              "gpu_fraction": 0.3,
              "optimizer": {
                "grads_per_step": 10
              },
            },
            #"trial_resources": {"cpu": 1, "gpu": 0, "extra_gpu": max(self.num_total_gpus-1, 1), "extra_cpu": 0},
            #"trial_resources": {"cpu": 1, "gpu": 0, "extra_gpu": max(self.num_total_gpus-1, 0),
            #                    "extra_cpu": max(self.num_cpus-1, 1)},
            "trial_resources": {"cpu": 1,
                                "extra_cpu": max(self.num_cpus-1, 1)},              
          }
        }

if __name__ == "__main__":
    os.environ["LC_ALL"] = "C.UTF-8"
    os.environ["LANG"] = "C.UTF-8"
    os.environ["RAY_USE_XRAY"] = "1"
    print(ppo.DEFAULT_CONFIG)
    MyLauncher().train_main()
```

---

**The Jupyter notebook:**

```
!/bin/bash ./setup.sh

from time import gmtime, strftime
import sagemaker
role = sagemaker.get_execution_role()

sage_session = sagemaker.session.Session()
s3_bucket = sage_session.default_bucket()  
s3_output_path = 's3://{}/'.format(s3_bucket)
print("S3 bucket path: {}".format(s3_output_path))

job_name_prefix = 'ArrivalSim'

from sagemaker.rl import RLEstimator, RLToolkit, RLFramework

estimator = RLEstimator(entry_point="mod_op_train.py", # Our launcher code
                        source_dir='src', # Directory where the supporting files are at. All of this will be
                                          # copied into the container.
                        dependencies=["common/sagemaker_rl"], # some other utils files.
                        toolkit=RLToolkit.RAY, # We want to run using the Ray toolkit against the ray container image.
                        framework=RLFramework.TENSORFLOW, # The code is in tensorflow backend.
                        toolkit_version='0.5.3', # Toolkit version. This will also choose an apporpriate tf version.                                               
                        #toolkit_version='0.6.5', # Toolkit version. This will also choose an apporpriate tf version.                        
                        role=role, # The IAM role that we created at the begining.
                        #train_instance_type="ml.m4.xlarge", # Since we want to run fast, lets run on GPUs.
                        train_instance_type="local", # Since we want to run fast, lets run on GPUs.
                        train_instance_count=1, # Single instance will also work, but running distributed makes things
                                                # fast, particularly in the case of multiple rollout training.
                        output_path=s3_output_path, # The path where we can expect our trained model.
                        base_job_name=job_name_prefix, # This is the name we setup above to be to track our job.
                        hyperparameters = {      # Some hyperparameters for Ray toolkit to operate.
                          "s3_bucket": s3_bucket,
                          "rl.training.stop.training_iteration": 2, # Number of iterations.
                          "rl.training.checkpoint_freq": 2,
                        },
                        #metric_definitions=metric_definitions, # This will bring all the logs out into the notebook.
                    )

estimator.fit()
```

---

See related [issue/motivation](https://stackoverflow.com/questions/57724414/how-to-make-the-inputs-and-model-have-the-same-shape-rllib-ray-sagemaker-reinfo/57762933#57762933).

---

<br>
