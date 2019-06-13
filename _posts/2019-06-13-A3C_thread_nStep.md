---
layout: posts
author: Huan
title: A3C multi-threaded version with N step targets

---
This post demonstrates how to implement the A3C (Asynchronous Advantage Actor Critic) algorithm with Tensorflow. This is a multi-threaded version.

N-step returns are used as critic's targets.

2 versions of N-step targets could be used:

Version 1) missing terms are treated as 0.

Version 2) use maximum terms possible.

Check this [post](https://chuacheowhuan.github.io/n_step_targets/) for more information on N-step targets.

Environment from OpenAI's gym: CartPole-v0 (Discrete)

[Full code](https://): A3C (discrete) multi-threaded version with version 1 of N-step targets

[Full code](https://): A3C (discrete) multi-threaded version with version 2 of N-step targets

Environment from OpenAI's gym: Pendulum-v0 (Continuous)

[Full code](https://): A3C (Continuous) multi-threaded version with version 1 of N-step targets

[Full code](https://): A3C (Continuous) multi-threaded version with version 2 of N-step targets

The ACNet class defines the models (Tensorflow graphs) and contains both the actor and the critic networks.

The Worker class contains the work function that does the main bulk of the computation.

A copy of ACNet is declared globally & the parameters are shared by the threaded workers. Each worker also have it's own local copy of ACNet.

ACNet class:

The following code segment describes the loss function for the actor & critic networks for the discrete environment:

```
TD_err = tf.subtract(self.critic_target, self.V, name='TD_err')
with tf.name_scope('actor_loss'):
    log_prob = tf.reduce_sum(tf.log(self.action_prob + 1e-5) * tf.one_hot(self.a, num_actions, dtype=tf.float32), axis=1, keep_dims=True)
    actor_component = log_prob * tf.stop_gradient(self.baselined_returns)
    # entropy for exploration
    entropy = -tf.reduce_sum(self.action_prob * tf.log(self.action_prob + 1e-5), axis=1, keep_dims=True)  # encourage exploration
    self.actor_loss = tf.reduce_mean( -(ENTROPY_BETA * entropy + actor_component) )                                        
with tf.name_scope('critic_loss'):
    self.critic_loss = tf.reduce_mean(tf.square(TD_err))
```

The following function in the ACNet class creates the actor and critic's neural networks:

```
def _create_net(self, scope):
    w_init = tf.glorot_uniform_initializer()
    with tf.variable_scope('actor'):
        hidden = tf.layers.dense(self.s, actor_hidden, tf.nn.relu6, kernel_initializer=w_init, name='hidden')
        action_prob = tf.layers.dense(hidden, num_actions, tf.nn.softmax, kernel_initializer=w_init, name='action_prob')        
    with tf.variable_scope('critic'):
        hidden = tf.layers.dense(self.s, critic_hidden, tf.nn.relu6, kernel_initializer=w_init, name='hidden')
        V = tf.layers.dense(hidden, 1, kernel_initializer=w_init, name='V')         
    actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
    critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')       
    return action_prob, V, actor_params, critic_params
```

Worker class:

The following code segment accumulates gradients & apply them to the local critic network:

```
self.AC.accumu_grad_critic(feed_dict) # accumulating gradients for local critic  
self.AC.apply_accumu_grad_critic(feed_dict)
```

The following code segment computes the advantage function:

```
baseline = SESS.run(self.AC.V, {self.AC.s: buffer_s}) # Value function
epr = np.vstack(buffer_r).astype(np.float32)
n_step_targets = self.compute_n_step_targets_missing(epr, baseline, GAMMA, N_step) # Q values
# Advantage function
baselined_returns = n_step_targets - baseline
```

The following code segment accumulates gradients for the local actor network:

```          
self.AC.accumu_grad_actor(feed_dict) # accumulating gradients for local actor  
```

The following code segment push the parameters from the local networks to the global networks and then pulls the updated global parameters to the local networks:

```
# update
self.AC.push_global_actor(feed_dict)                
self.AC.push_global_critic(feed_dict)
buffer_s, buffer_a, buffer_r, buffer_done = [], [], [], []
self.AC.pull_global()
```

The following code segment initialize storage for accumulated local gradients.

```
self.AC.init_grad_storage_actor() # initialize storage for accumulated gradients.
self.AC.init_grad_storage_critic()            
```
