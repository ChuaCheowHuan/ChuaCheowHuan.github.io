---
layout: posts
author: Huan
title: Dueling DDQN with PER
---

This post documents my implementation of the Dueling Double Deep Q Network with Priority Experience Replay (Duel DDQN with PER) algorithm.

---

A **Dueling Double Deep Q Network with Priority Experience Replay
(Duel DDQN with PER)** implementation in tensorflow. The code is tested with
Gym's discrete action space environment, CartPole-v0 on Colab.

Code on my [Github](https://github.com/ChuaCheowHuan/reinforcement_learning/tree/master/DQN_variants/duel_DDQN_PER/duelling_DDQN_PER_cartpole.ipynb)

If Github is not loading the Jupyter notebook, a known Github issue, click [here](https://nbviewer.jupyter.org/github/ChuaCheowHuan/reinforcement_learning/blob/master/DQN_variants/duel_DDQN_PER/duelling_DDQN_PER_cartpole.ipynb)
to view the notebook on Jupyter's nbviewer.

---

## Notations:

Model network = $$Q_{\theta}$$

Model parameter = $$\theta$$

Model network Q value = $$Q_{\theta}$$ (s, a)

Target network = $$Q_{\phi}$$

Target parameter = $$\phi$$

Target network Q value = $$Q_{\phi}$$ ($$s^{'}$$, $$a^{'}$$)

A small constant to ensure that no sample has 0 probability to be selected = e

Hyper parameter  = $$\alpha$$

- Decides how to sample, range from 0 to 1, where 0 corresponds to fully
uniformly random sample selection & 1 corresponding to selecting samples based
on highest priority.  

Hyper parameter  = $$\beta$$

- Starts close to 0, gradually annealed  to 1, slowly giving more importance to weights during training.

Minibatch size = k

Replay memory size = N

---

## Equations:

TD target =
r (s, a)
$$+$$
$$\gamma$$
$$Q_{\phi}$$
($$s^{'}$$,
$$argmax_{a^{'}}$$
$$Q_{\theta}$$
(s$$^{'}$$, a$$^{'}$$))

TD error =
$${\delta}$$
=
(TD target)
$$-$$
(Model network Q value)
=
[r (s, a)
$$+$$
$$\gamma$$
$$Q_{\phi}$$
($$s^{'}$$,
$$argmax_{a^{'}}$$
$$Q_{\theta}$$
(s$$^{'}$$,
a$$^{'}$$))]
$$-$$
$$Q_{\theta}$$
(s, a)

$$priority_{i}$$ =
$$p_{i}$$
=
$${|\delta_{i}|}$$ $$+$$ e

probability(i) =
P(i)
=
$$\frac{p_{i}^{\alpha}}  
{\sum_{k}p_{k}^{\alpha}}$$

weights =
$$w_{i}$$
=
(N $$\cdot$$ P(i))
$$^{-\beta}$$

---

## Key implementation details:

**Sum tree:**

Assume an example of a sum tree with 7 nodes (with 4 leaves which corresponds to the replay memory size):

At initialization:
- ![alt text](https://raw.githubusercontent.com/ChuaCheowHuan/reinforcement_learning/master/DQN_variants/duel_DDQN_PER/img/pic_1.png)

When item 1 is added:
- ![alt text](https://raw.githubusercontent.com/ChuaCheowHuan/reinforcement_learning/master/DQN_variants/duel_DDQN_PER/img/pic_2.png)

When item 2 is added:
- ![alt text](https://raw.githubusercontent.com/ChuaCheowHuan/reinforcement_learning/master/DQN_variants/duel_DDQN_PER/img/pic_3.png)

When item 3 is added:
- ![alt text](https://raw.githubusercontent.com/ChuaCheowHuan/reinforcement_learning/master/DQN_variants/duel_DDQN_PER/img/pic_4.png)

When item 4 is added:
- ![alt text](https://raw.githubusercontent.com/ChuaCheowHuan/reinforcement_learning/master/DQN_variants/duel_DDQN_PER/img/pic_5.png)

When item 5 is added:
- ![alt text](https://raw.githubusercontent.com/ChuaCheowHuan/reinforcement_learning/master/DQN_variants/duel_DDQN_PER/img/pic_6.png)

Figure below shows the corresponding code & array contents. The tree represents the entire sum tree while data represents the leaves.

![alt text](https://raw.githubusercontent.com/ChuaCheowHuan/reinforcement_learning/master/DQN_variants/duel_DDQN_PER/img/pic_7.png)

In the implementation, only one sumTree object is needed to store the collected experiences, this sumTree object resides in the Replay_memory class. The sumTree object has number of leaves = replay memory size = capacity.
The data array in sumTree object stores an Exp object, which is a sample of experience.

The following code decides how to sample:

```
def sample(self, k): # k = minibatch size
    batch = []

    # total_p() gives the total sum of priorities of the leaves in the sumTree
    # which is the value stored in the root node
    segment = self.tree.total_p() / k

    for i in range(k):
        a = segment * i # start of segment
        b = segment * (i + 1) # end of segment
        s = np.random.uniform(a, b) # rand value between a, b

        (idx, p, data) = self.tree.get(s)
        batch.append( (idx, p, data) )            

    return batch    
```

Refer to appendix B.2.1, under the section, "Proportional prioritization", from the original (Schaul et al., 2016) [paper](https://arxiv.org/pdf/1511.05952.pdf) for sampling details.

---

## Tensorflow graph:

![image](/assets/images/DQN_variants_tf_graph_img/Duel_DDQN_PER_tf_graph.png)

---

## References:

[Prioritized experience replay (Schaul et al., 2016)](https://arxiv.org/pdf/1511.05952.pdf)

---

<br>
