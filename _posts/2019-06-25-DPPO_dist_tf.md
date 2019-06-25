---
layout: posts
author: Huan
title: DPPO distributed tensorflow
---

DPPO continuous (normalized running rewards with GAE) implementation with distributed Tensorflow and Python's multiprocessing package. This is a continuous version.

Environment from OpenAI's gym: Pendulum-v0 (Continuous)

[Full code](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/DPPO/DPPO_cont_GAE_dist_GPU.ipynb):

**Notations:**

current policy = $${\pi}_{\theta} (a_{t} {\mid} s_{t})$$

old policy = $${\pi}_{\theta_{old}} (a_{t} {\mid} s_{t})$$

epsilon = $${\epsilon}$$

Advantage function = A

---
<br>

**Equations:**

Truncated version of generalized advantage estimation(GAE) =

A_{t} =

when lamda = 1,

A_{t} =

Probability ratio =

$$R_{t}({\theta})$$ = $${\dfrac{ {\pi}_{\theta} (a_{t} {\mid} s_{t}) } { {\pi}_{\theta_{old}} (a_{t} {\mid} s_{t}) } }$$

Objective function =

$$
L^{CLIP}
({\theta})
$$
=
$$
\mathop{\mathbb{E_{t}}}
\lbrack
min(
  R_{t}({\theta})
  A_{t}
  ,
  clip
  (
    R_{t}({\theta}),
    1+{\epsilon},
    1-{\epsilon}
    )
    A_{t}
  )
\rbrack
$$

---
<br>

**Implementation details:**

The following code segment from the PPO class defines the Clipped Surrogate Objective function:

```
with tf.variable_scope('surrogate'):
                    ratio = self.pi.prob(self.act) / self.oldpi.prob(self.act)
                    surr = ratio * self.adv
                    self.aloss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1.-epsilon, 1.+epsilon)*self.adv))
```

The following code segment from the work() function in the worker class normalized the running rewards for each worker:

```
self.running_stats_r.update(np.array(buffer_r))
                    buffer_r = np.clip( (np.array(buffer_r) - self.running_stats_r.mean) / self.running_stats_r.std, -stats_CLIP, stats_CLIP )
```

The following code segment from the work() function in the worker class computes the TD lamda return & advantage:

```
tdlamret, adv = self.ppo.add_vtarg_and_adv(np.vstack(buffer_r), np.vstack(buffer_done), np.vstack(buffer_V), v_s_, GAMMA, lamda)

```

The following update function in the PPO class does the training & the updating of global & local parameters
(Note the at the beginning of training, probability ratio = 1):

```
def update(self, s, a, r, adv):    
    self.sess.run(self.update_oldpi_op)

    for _ in range(A_EPOCH): # train actor
        self.sess.run(self.atrain_op, {self.state: s, self.act: a, self.adv: adv})
        # update actor
        self.sess.run([self.push_actor_pi_params,
                       self.pull_actor_pi_params],
                      {self.state: s, self.act: a, self.adv: adv})
    for _ in range(C_EPOCH): # train critic
        # update critic
        self.sess.run(self.ctrain_op, {self.state: s, self.discounted_r: r})
        self.sess.run([self.push_critic_params,
                       self.pull_critic_params],
                      {self.state: s, self.discounted_r: r})   
```

The distributed Tensorflow & multiprocessing code sections are very similar to the ones describe in the following posts:

[Distributed Tensorflow](https://chuacheowhuan.github.io/dist_tf/)

[A3C distributed tensorflow](https://chuacheowhuan.github.io/A3C_dist_tf/)

---
<br>

**References:**

[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)
(Schulman, Wolski, Dhariwal, Radford, Klimov, 2017)

[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf)
(Nicolas Heess, Dhruva TB, Srinivasan Sriram, Jay Lemmon, Josh Merel, Greg Wayne, et al., 2017)
