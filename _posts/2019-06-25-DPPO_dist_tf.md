---
layout: posts
author: Huan
title: DPPO distributed tensorflow
---

Distributed Proximal Policy Optimization (Distributed PPO or DPPO) continuous
version implementation with distributed Tensorflow and Python's multiprocessing
package. This implementation uses normalized running rewards with GAE. The code
is tested with Gym's continuous action space environment, Pendulum-v0 on Colab.

[Full code](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/DPPO/DPPO_cont_GAE_dist_GPU.ipynb):

---

## Notations:

current policy =
$${\pi}_{\theta}
(a_{t}
  {\mid} s_{t})$$

old policy =
$${\pi}_{\theta_{old}}
(a_{t}
  {\mid} s_{t})$$

epsilon =
$${\epsilon}$$

Advantage function = A

---

## Equations:

Truncated version of generalized advantage estimation (GAE) =

$$
A_{t}
$$
=
$$
{\delta}_{t}
+
({\gamma}
{\lambda})
{\delta}_{t}
+
...
+
({\gamma}
{\lambda})
^{T-t+1}
{\delta}_{T-1}
$$

where
$${\delta}_{t}$$ =
$$
{r}_{t} +
{\gamma}
V(s_{t+1}) -
V(s_{t})
$$

when $${\lambda}$$ = 1,

$$A_{t}$$ =
$$
-V(s_{t}) +
r_{t} +
{\gamma}r_{t+1} +
... +
{\gamma}^{T-t+1}
r_{T-1} +
{\gamma}^{T-t}
V(s_{T})
$$

Probability ratio =

$$R_{t}({\theta})$$ = $${\dfrac{ {\pi}_{\theta} (a_{t} {\mid} s_{t}) } { {\pi}_{\theta_{old}} (a_{t} {\mid} s_{t}) } }$$

Clipped Surrogate Objective function =

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

## Key implementation details:

The following class is adapted from OpenAI's baseline:
This class is used for the normalization of rewards in this program before GAE
computation.

```
class RunningStats(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.std = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.std = np.maximum(np.sqrt(self.var), 1e-6)
        self.count = batch_count + self.count
```

This function in the ```PPO``` class is adapted from OpenAI's Baseline,
returns TD lamda return & advantage

```
    def add_vtarg_and_adv(self, R, done, V, v_s_, gamma, lam):
        # Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        # last element is only used for last vtarg, but we already zeroed it if last new = 1
        done = np.append(done, 0)
        V_plus = np.append(V, v_s_)
        T = len(R)
        adv = gaelam = np.empty(T, 'float32')
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-done[t+1]        
            delta = R[t] + gamma * V_plus[t+1] * nonterminal - V_plus[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam   
        #print("adv=", adv.shape)
        #print("V=", V.shape)
        #print("V_plus=", V_plus.shape)
        tdlamret = np.vstack(adv) + V
        #print("tdlamret=", tdlamret.shape)
        return tdlamret, adv # tdlamret is critic_target or Qs      
```

The following code segment from the ```PPO``` class defines the Clipped Surrogate
Objective function:

```
with tf.variable_scope('surrogate'):
                    ratio = self.pi.prob(self.act) / self.oldpi.prob(self.act)
                    surr = ratio * self.adv
                    self.aloss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1.-epsilon, 1.+epsilon)*self.adv))
```

The following code segment from the ```work()``` function in the worker class
normalized the running rewards for each worker:

```
self.running_stats_r.update(np.array(buffer_r))
                    buffer_r = np.clip( (np.array(buffer_r) - self.running_stats_r.mean) / self.running_stats_r.std, -stats_CLIP, stats_CLIP )
```

The following code segment from the ```work()``` function in the worker class computes
 the TD lamda return & advantage:

```
tdlamret, adv = self.ppo.add_vtarg_and_adv(np.vstack(buffer_r), np.vstack(buffer_done), np.vstack(buffer_V), v_s_, GAMMA, lamda)

```

The following update function in the ```PPO``` class does the training & the
updating of global & local parameters (Note the at the beginning of training,
  probability ratio = 1):

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

---

The distributed Tensorflow & multiprocessing code sections are very similar to
the ones describe in the following posts:

[A3C distributed tensorflow](https://chuacheowhuan.github.io/A3C_dist_tf/)

[Distributed Tensorflow](https://chuacheowhuan.github.io/dist_tf/)

---

## References:

[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)
(Schulman, Wolski, Dhariwal, Radford, Klimov, 2017)

[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf)
(Nicolas Heess, Dhruva TB, Srinivasan Sriram, Jay Lemmon, Josh Merel, Greg Wayne, et al., 2017)

---

<br>
