---
layout: posts
author: Huan
title: N-step targets

---
N-step Q-values estimation.

[Full code](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/misc_python_examples/n_step_targets.ipynb)

The following two functions computes truncated Q-values estimates:

1) n_step_targets_missing

treats missing terms as 0.

2) n_step_targets_max

use maximum terms possible.

<br>

1-step truncated estimate :

$$Q^{\pi}(s_{t}, a_{t})$$ = E($$r_{t}$$ + $$\gamma$$ V($$s_{t+1}$$))

2-step truncated estimate :

$$Q^{\pi}(s_{t}, a_{t})$$ = E($$r_{t}$$ + $$\gamma$$ $$r_{t+1}$$ +  $$\gamma^{2}$$ V($$s_{t+2}$$))

3-step truncated estimate :

$$Q^{\pi}(s_{t}, a_{t})$$ = E($$r_{t}$$ + $$\gamma$$ $$r_{t+1}$$ + $$\gamma^{2}$$ $$r_{t+2}$$ + $$\gamma^{3}$$ V($$s_{t+3}$$))

N-step truncated estimate :

$$Q^{\pi}(s_{t}, a_{t})$$ = E($$r_{t}$$ + $$\gamma$$ $$r_{t+1}$$ + $$\gamma^{2}$$ $$r_{t+2}$$ + ... + $$\gamma^{n}$$ V($$s_{t+n}$$))

<br>

Assuming we have the following variables setup:
```
N=2 # N steps
gamma=2
t=5
v_s_ = 10 # value of next state

epr=np.arange(t).reshape(t,1)
print("epr=", epr)

baselines=np.arange(t).reshape(t,1)
print("baselines=", baselines)
```

Display output of episodic rewards(epr) & baselines:
```
epr= [[0]
 [1]
 [2]
 [3]
 [4]]

baselines= [[0]
 [1]
 [2]
 [3]
 [4]]
```

This function computes the n-step targets, treats missing terms as zero:
```
# if number of steps unavailable, missing terms treated as 0.
def n_step_targets_missing(epr, baselines, gamma, N):
  N = N+1
  targets = np.zeros_like(epr)    
  if N > epr.size:
    N = epr.size
  for t in range(epr.size):   
    print("t=", t)
    for n in range(N):
      print("n=", n)
      if t+n == epr.size:            
        print('missing terms treated as 0, break') # last term for those with insufficient steps.
        break # missing terms treated as 0
      if n == N-1: # last term
        targets[t] += (gamma**n) * baselines[t+n] # last term for those with sufficient steps
        print('last term for those with sufficient steps, end inner n loop')
      else:
        targets[t] += (gamma**n) * epr[t+n] # non last terms
  return targets
```

Run the function n_step_targets_missing:
```
print('n_step_targets_missing:')
T = n_step_targets_missing(epr, baselines, gamma, N)
print(T)
```

Display the output:
```
n_step_targets_missing:
t= 0
n= 0
n= 1
n= 2
last term for those with sufficient steps, end inner n loop
t= 1
n= 0
n= 1
n= 2
last term for those with sufficient steps, end inner n loop
t= 2
n= 0
n= 1
n= 2
last term for those with sufficient steps, end inner n loop
t= 3
n= 0
n= 1
n= 2
missing terms treated as 0, break
t= 4
n= 0
n= 1
missing terms treated as 0, break
[[10]
 [17]
 [24]
 [11]
 [ 4]]
```
For the output above, note that when t+n = 5 which is greater than the last index 4, missing terms are treated as 0.

<br>

This function computes the n-step targets, it will use maximum number of terms possible:
```
# if number of steps unavailable, use max steps available.
# uses v_s_ as input
def n_step_targets_max(epr, baselines, v_s_, gamma, N):
  N = N+1
  targets = np.zeros_like(epr)    
  if N > epr.size:
    N = epr.size
  for t in range(epr.size):  
    print("t=", t)
    for n in range(N):
      print("n=", n)
      if t+n == epr.size:            
        targets[t] += (gamma**n) * v_s_ # last term for those with insufficient steps.
        print('last term for those with INSUFFICIENT steps, break')
        break
      if n == N-1:
        targets[t] += (gamma**n) * baselines[t+n] # last term for those with sufficient steps
        print('last term for those with sufficient steps, end inner n loop')
      else:
        targets[t] += (gamma**n) * epr[t+n] # non last terms
  return targets
```

Run the function n_step_targets_max:
```
print('n_step_targets_max:')
T = n_step_targets_max(epr, baselines, v_s_, gamma, N)
print(T)
```

Display the output:
```
n_step_targets_max:
t= 0
n= 0
n= 1
n= 2
last term for those with sufficient steps, end inner n loop
t= 1
n= 0
n= 1
n= 2
last term for those with sufficient steps, end inner n loop
t= 2
n= 0
n= 1
n= 2
last term for those with sufficient steps, end inner n loop
t= 3
n= 0
n= 1
n= 2
last term for those with INSUFFICIENT steps, break
t= 4
n= 0
n= 1
last term for those with INSUFFICIENT steps, break
[[10]
 [17]
 [24]
 [51]
 [24]]
```
For the output above, note that when t+n = 5 which is greater than the last index 4, maximum terms are used where possible. ( Last term for those with INSUFFICIENT steps is given by (gamma**n) * v_s_ = $$\gamma^{5}$$ V($$s_{5}$$)), where v_s_ = V($$s_{5}$$)

t=2, normal 2 steps estimation:

$$Q^{\pi}(s_{t}, a_{t})$$ = E($$r_{2}$$ + $$\gamma$$ $$r_{3}$$ +  $$\gamma^{4}$$ V($$s_{4}$$))

t=3, 2 steps estimation with insufficient step, using v_s_ in the last term:

$$Q^{\pi}(s_{t}, a_{t})$$ = E($$r_{3}$$ + $$\gamma$$ $$r_{4}$$ +  $$\gamma^{5}$$ V($$s_{5}$$))

t=4, insufficient step for 2 steps estimation, resorting to 1 step estimation:

$$Q^{\pi}(s_{t}, a_{t})$$ = E($$r_{4}$$ + $$\gamma^{5}$$ V($$s_{5}$$))
