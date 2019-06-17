---
layout: posts
author: Huan
title: A3C multi-threaded continuous version with N step targets

---
pending update...

An A3C (Asynchronous Advantage Actor Critic) algorithm implementation with
Tensorflow. This is a multi-threaded continuous version.

Environment from OpenAI's gym: Pendulum-v0 (Discrete)

[Full code](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/policy_gradient_based/A3C/A3C_cont_max.ipynb): A3C (continuous) multi-threaded version with N-step
targets(use maximum terms possible)

The majority of the code is very similar to the [discrete](https://chuacheowhuan.github.io/A3C_disc_thread_nStep/) version with the
exceptions highlighted in the following sections:

Action selection:

```
with tf.name_scope('select_action'):
    #mean = mean * action_bound[1]                   
    mean = mean * ( action_bound[1] - action_bound[0] ) / 2
    sigma += 1e-4
    normal_dist = tf.distributions.Normal(mean, sigma)                     
    self.choose_a = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), action_bound[0], action_bound[1])                  
```

Loss function of the actor network:

```
with tf.name_scope('actor_loss'):
    log_prob = normal_dist.log_prob(self.a)
    #actor_component = log_prob * tf.stop_gradient(TD_err)
    actor_component = log_prob * tf.stop_gradient(self.baselined_returns)
    entropy = -tf.reduce_mean(normal_dist.entropy()) # Compute the differential entropy of the multivariate normal.                   
    self.actor_loss = -tf.reduce_mean( ENTROPY_BETA * entropy + actor_component)
```

The following code segment creates a LSTM layer:

```
def _lstm(self, Inputs, cell_size):
        # [time_step, feature] => [time_step, batch, feature]
        s = tf.expand_dims(Inputs, axis=1, name='time_major')  
        lstm_cell = tf.nn.rnn_cell.LSTMCell(cell_size)
        self.init_state = lstm_cell.zero_state(batch_size=1, dtype=tf.float32)
        outputs, self.final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=s, initial_state=self.init_state, time_major=True)
        # joined state representation          
        lstm_out = tf.reshape(outputs, [-1, cell_size], name='flatten_rnn_outputs')  
        return lstm_out
```

The following function in the ACNet class creates the actor and critic's neural networks(note that the critic's network contains a LSTM layer):

```
def _create_net(self, scope):
    w_init = tf.glorot_uniform_initializer()
    #w_init = tf.random_normal_initializer(0., .1)
    with tf.variable_scope('actor'):                        
        hidden = tf.layers.dense(self.s, actor_hidden, tf.nn.relu6, kernel_initializer=w_init, name='hidden')            
        #lstm_out = self._lstm(hidden, cell_size)
        # tanh range = [-1,1]
        mean = tf.layers.dense(hidden, num_actions, tf.nn.tanh, kernel_initializer=w_init, name='mean')
        # softplus range = {0,inf}
        sigma = tf.layers.dense(hidden, num_actions, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
    with tf.variable_scope('critic'):
        hidden = tf.layers.dense(self.s, critic_hidden, tf.nn.relu6, kernel_initializer=w_init, name='hidden')
        lstm_out = self._lstm(hidden, cell_size)
        V = tf.layers.dense(lstm_out, 1, kernel_initializer=w_init, name='V')  
    actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
    critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
    return mean, sigma, V, actor_params, critic_params
```
