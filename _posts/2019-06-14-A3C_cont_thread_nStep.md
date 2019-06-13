---
layout: posts
author: Huan
title: A3C multi-threaded continuous version with N step targets

---
This post demonstrates how to implement the A3C (Asynchronous Advantage Actor Critic) algorithm with Tensorflow. This is a multi-threaded continuous version.

N-step returns are used as critic's targets.

2 versions of N-step targets could be used:

Version 1) missing terms are treated as 0.

Version 2) use maximum terms possible.

Check this [post](https://chuacheowhuan.github.io/n_step_targets/) for more information on N-step targets.

Environment from OpenAI's gym: Pendulum-v0 (Continuous)

[Full code](https://): A3C (Continuous) multi-threaded version with version 1 of N-step targets

[Full code](https://): A3C (Continuous) multi-threaded version with version 2 of N-step targets



ACNet class:

The following code segment describes the loss function for the actor & critic networks for the discrete environment:

```

```

Worker class:
