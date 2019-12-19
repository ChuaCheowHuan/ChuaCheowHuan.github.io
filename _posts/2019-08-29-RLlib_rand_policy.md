---
layout: posts
author: Huan
title: Random policy in RLlib
---

Creating & seeding a random policy class in RLlib.

---

Code on my [Github](https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction/blob/master/gym_continuousDoubleAuction/CDA_env_disc_RLlib.py)

---

**Function:**

```
def make_RandomPolicy(_seed):

    # a hand-coded policy that acts at random in the env (doesn't learn)
    class RandomPolicy(Policy):
        """Hand-coded policy that returns random actions."""
        def __init__(self, observation_space, action_space, config):
            self.observation_space = observation_space
            self.action_space = action_space
            self.action_space.seed(_seed)

        def compute_actions(self,
                            obs_batch,
                            state_batches,
                            prev_action_batch=None,
                            prev_reward_batch=None,
                            info_batch=None,
                            episodes=None,
                            **kwargs):
            """Compute actions on a batch of observations."""
            return [self.action_space.sample() for _ in obs_batch], [], {}

        def learn_on_batch(self, samples):
            """No learning."""
            #return {}
            pass

        def get_weights(self):
            pass

        def set_weights(self, weights):
            pass

    return RandomPolicy
```

---

**Usage example:**

```
# Setup PPO with an ensemble of `num_policies` different policies
    policies = {"policy_{}".format(i): gen_policy(i) for i in range(args.num_policies)} # contains many "policy_graphs" in a policies dictionary

    # override policy with random policy
    policies["policy_{}".format(args.num_policies-3)] = (make_RandomPolicy(1), obs_space, act_space, {}) # random policy stored as the last item in policies dictionary
    policies["policy_{}".format(args.num_policies-2)] = (make_RandomPolicy(2), obs_space, act_space, {}) # random policy stored as the last item in policies dictionary
    policies["policy_{}".format(args.num_policies-1)] = (make_RandomPolicy(3), obs_space, act_space, {}) # random policy stored as the last item in policies dictionary
```
