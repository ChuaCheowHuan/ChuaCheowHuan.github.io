---
layout: posts
author: Huan
title: Accumulate gradients with Tensorflow

---
This post demonstrates how to accumulate gradients with Tensorflow.

[Full code](https://github.com/ChuaCheowHuan/misc_code_examples/blob/master/tf/tf_accumulate_grad.ipynb)

```
import tensorflow as tf

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
```
