---
layout: posts
author: Huan
title: A3C (discrete) multi-threaded version with N step targets

---
This post demonstrates how to implement the A3C (Asynchronous Advantage Actor Critic) algorithm with Tensorflow.

This is a multi-threaded version which learns in a discrete environment.

N-step returns are used as critic's targets.

2 versions of N-step targets could be used:

1) missing terms are treated as 0.

2) use maximum terms possible.

Check this [post](https://chuacheowhuan.github.io/n_step_targets/) for more information on N-step targets.

Environment from OpenAI's gym: CartPole-v0

[Full code](https://)

ACNet class

```
class ACNet(object):
    def __init__(self, scope, globalAC=None):   
        if scope == net_scope: # global
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, num_obvs], 'S')
                # create global net
                self.actor_params, self.critic_params = self._create_net(scope)[-2:] # only require params

        else: # local
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, num_obvs], 'S')
                self.a = tf.placeholder(tf.int32, [None, ], 'A')
                self.critic_target = tf.placeholder(tf.float32, [None, 1], 'critic_target')
                self.baselined_returns = tf.placeholder(tf.float32, [None, 1], 'baselined_returns') # for calculating advantage
                # create local net
                self.action_prob, self.V, self.actor_params, self.critic_params = self._create_net(scope)

                TD_err = tf.subtract(self.critic_target, self.V, name='TD_err')
                with tf.name_scope('actor_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.action_prob + 1e-5) * tf.one_hot(self.a, num_actions, dtype=tf.float32), axis=1, keep_dims=True)
                    actor_component = log_prob * tf.stop_gradient(self.baselined_returns)
                    # entropy for exploration
                    entropy = -tf.reduce_sum(self.action_prob * tf.log(self.action_prob + 1e-5), axis=1, keep_dims=True)  # encourage exploration
                    self.actor_loss = tf.reduce_mean( -(ENTROPY_BETA * entropy + actor_component) )                                        
                with tf.name_scope('critic_loss'):
                    self.critic_loss = tf.reduce_mean(tf.square(TD_err))                      
                # accumulated gradients for local actor    
                with tf.name_scope('local_actor_grad'):                   
                    self.actor_zero_op, self.actor_accumu_op, self.actor_apply_op, actor_accum = self.accumu_grad(OPT_A, self.actor_loss, scope=scope + '/actor')
                # accumulated gradients for local critic    
                with tf.name_scope('local_critic_grad'):
                    self.critic_zero_op, self.critic_accumu_op, self.critic_apply_op, critic_accum = self.accumu_grad(OPT_C, self.critic_loss, scope=scope + '/critic')

            with tf.name_scope('params'): # push/pull from local/worker perspective
                with tf.name_scope('push_to_global'):
                    self.push_actor_params = OPT_A.apply_gradients(zip(actor_accum, globalAC.actor_params))
                    self.push_critic_params = OPT_C.apply_gradients(zip(critic_accum, globalAC.critic_params))
                with tf.name_scope('pull_fr_global'):
                    self.pull_actor_params = [local_params.assign(global_params) for local_params, global_params in zip(self.actor_params, globalAC.actor_params)]
                    self.pull_critic_params = [local_params.assign(global_params) for local_params, global_params in zip(self.critic_params, globalAC.critic_params)]                    

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

    def accumu_grad(self, OPT, loss, scope):
        # retrieve trainable variables in scope of graph
        #tvs = tf.trainable_variables(scope=scope + '/actor')
        tvs = tf.trainable_variables(scope=scope)
        # ceate a list of variables with the same shape as the trainable
        accumu = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        zero_op = [tv.assign(tf.zeros_like(tv)) for tv in accumu] # initialized with 0s
        gvs = OPT.compute_gradients(loss, tvs) # obtain list of gradients & variables
        #gvs = [(tf.where( tf.is_nan(grad), tf.zeros_like(grad), grad ), var) for grad, var in gvs]
        # adds to each element from the list you initialized earlier with zeros its gradient
        # accumu and gvs are in same shape, index 0 is grads, index 1 is vars
        accumu_op = [accumu[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
        apply_op = OPT.apply_gradients([(accumu[i], gv[1]) for i, gv in enumerate(gvs)]) # apply grads
        return zero_op, accumu_op, apply_op, accumu      

    def push_global_actor(self, feed_dict):  
        SESS.run([self.push_actor_params], feed_dict)  

    def push_global_critic(self, feed_dict):  
        SESS.run([self.push_critic_params], feed_dict)         

    def pull_global(self):  
        SESS.run([self.pull_actor_params, self.pull_critic_params])

    def choose_action(self, s):  
        prob_weights = SESS.run(self.action_prob, feed_dict={self.s: s[None, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action             

    def init_grad_storage_actor(self):
        SESS.run(self.actor_zero_op)

    def accumu_grad_actor(self, feed_dict):
        SESS.run([self.actor_accumu_op], feed_dict)          

    def apply_accumu_grad_actor(self, feed_dict):
        SESS.run([self.actor_apply_op], feed_dict)   

    def init_grad_storage_critic(self):
        SESS.run(self.critic_zero_op)

    def accumu_grad_critic(self, feed_dict):
        SESS.run([self.critic_accumu_op], feed_dict)          

    def apply_accumu_grad_critic(self, feed_dict):
        SESS.run([self.critic_apply_op], feed_dict)  
```

Worker class

```
class Worker(object): # local only
    def __init__(self, name, globalAC):
        self.env = gym.make(game)
        self.name = name
        self.AC = ACNet(name, globalAC)
    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        T = 0
        t = 0
        while not COORD.should_stop() and GLOBAL_EP < max_global_episodes:
            s = self.env.reset()
            ep_r = 0 # reward per episode
            done = False
            buffer_s, buffer_a, buffer_r, buffer_done = [], [], [], []
            self.AC.pull_global()
            while not done:
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                buffer_done.append(done)                
                s = s_
                t += 1

            # if statement will always be done in this case...
            # possible future modification
            if done:
                V_s = 0   
            else:
                V_s = SESS.run(self.AC.V, {self.AC.s: s[None, :]})[0, 0] # takes in just one s, not a batch.

            # critic related
            critic_target = self.discount_rewards(buffer_r, GAMMA, V_s)

            buffer_s, buffer_a, critic_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(critic_target)
            feed_dict = {self.AC.s: buffer_s, self.AC.critic_target: critic_target}                         
            self.AC.accumu_grad_critic(feed_dict) # accumulating gradients for local critic  
            self.AC.apply_accumu_grad_critic(feed_dict)

            baseline = SESS.run(self.AC.V, {self.AC.s: buffer_s}) # Value function
            epr = np.vstack(buffer_r).astype(np.float32)
            n_step_targets = self.compute_n_step_targets_missing(epr, baseline, GAMMA, N_step) # Q values
            # Advantage function
            baselined_returns = n_step_targets - baseline

            feed_dict = {self.AC.s: buffer_s, self.AC.a: buffer_a, self.AC.critic_target: critic_target, self.AC.baselined_returns: baselined_returns}            
            self.AC.accumu_grad_actor(feed_dict) # accumulating gradients for local actor  

            # update
            self.AC.push_global_actor(feed_dict)                
            self.AC.push_global_critic(feed_dict)
            buffer_s, buffer_a, buffer_r, buffer_done = [], [], [], []
            self.AC.pull_global()

            if T % delay_rate == 0: # delay clearing of local gradients storage to reduce noise
                # apply to local
                self.AC.init_grad_storage_actor() # initialize storage for accumulated gradients.
                self.AC.init_grad_storage_critic()

            GLOBAL_RUNNING_R.append(ep_r) # for display
            GLOBAL_EP += 1                           

    def discount_rewards(self, r, gamma, running_add):
      """Take 1D float array of rewards and compute discounted reward """
      discounted_r = np.zeros_like(r)
      #running_add = 0
      for t in reversed(range(len(r))):
          running_add = running_add * gamma + r[t]
          discounted_r[t] = running_add
      return discounted_r

    # As n increase, variance increase.
    # Create a function that returns an array of n-step targets, one for each timestep:
    # target[t] = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... + \gamma^n V(s_{t+n})
    # Where r_t is given by episode reward (epr) and V(s_n) is given by the baselines.
    def compute_n_step_targets_missing(self, epr, baselines, gamma, N):
      targets = np.zeros_like(epr)    
      if N > epr.size:
        N = epr.size
      for t in range(epr.size):    
        for n in range(N):
          if t+n == epr.size:            
            break # missing terms treated as 0
          if n == N-1: # last term
            targets[t] += (gamma**n) * baselines[t+n]
          else:
            targets[t] += (gamma**n) * epr[t+n]
      return targets  

    def compute_n_step_targets_2(self, r, B, g, N):
      if N >= len(r):
        N = len(r)-1

      T = np.zeros_like(r)             

      # Non n-steps ops without baseline terms
      t = r.size-1
      T[t] = r[t] # last entry, do 0 step
      for n in range(1,N): # n = 1..N-1, do 1 step to N-1 step
        t = t-1
        for i in range(n): # get 0..n-1 gamma raised r terms
          T[t] += g**i * r[t+i]

      # Non n-steps ops with baseline terms
      t = r.size-1
      for j in range(1,N): # 1..N-1
        t = t-1
        T[t] += g**j * B[N]

      # n-steps ops without baseline
      for t in range(r.size-N): # 0..r.size-N-1
        for k in range(N):
          T[t] += g**k * r[t+k]

      # n-steps ops with baseline
      for t in range(r.size-N): # 0..r.size-N-1
        T[t] += g**N * B[t+N]

      return T
```

Output:

```
WARNING:tensorflow:From <ipython-input-2-cf49427a73a9>:45: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
Instructions for updating:
Use keras.layers.dense instead.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From <ipython-input-2-cf49427a73a9>:20: calling reduce_sum_v1 (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.

```

Output chart:

![alt text](data:image/png;base64,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)
