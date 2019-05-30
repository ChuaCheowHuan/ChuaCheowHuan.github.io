---
layout: posts
author: Huan
title: Numpy array manipulation

---

Simple numpy array manipulation examples

[Code](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/misc_python_examples/np_array_manipulation.ipynb)

This Jupyter notebook contains simple examples on how to manipulate numpy arrays. The code block below shows the codes & it's corresponding display output.

Setting up a numpy array:
```
buffer=[0,1]
print('buffer=', buffer)
$buffer= [0, 1]

new=2
print('new=', new)
$new= 2

buffer = np.array(buffer + [new]) # append a new item & create a numpy array
print('np.array(buffer + [new])=', buffer)
$np.array(buffer + [new])= [0 1 2]
```

Slicing examples:
```
# numpy array slicing syntax
# buffer[start:stop:step]

print('buffer[1:]=', buffer[1:]) # starting from index 1
$buffer[1:]= [1 2]

print('buffer[-1:]=', buffer[-1:]) # getting item in last index
$buffer[-1:]= [2]

print('buffer[:1]=', buffer[:1]) # stop at index 1 (exclusive), keep only 1st item
$buffer[:1]= [0]

print('buffer[:-1]=', buffer[:-1]) # stop at last index (exclusive), discard item in last index
$buffer[:-1]= [0 1]

print('buffer[::-1]=', buffer[::-1]) # start from last index (reversal)
$buffer[::-1]= [2 1 0]

print('buffer[1::-1]=', buffer[1::-1]) # reverse starting from index 1
$buffer[1::-1]= [1 0]

# Starting from index 1 will return [1 2], reversing will return [2,1]
print('buffer[1:][::-1]=', buffer[1:][::-1])
$buffer[1:][::-1]= [2 1]
```

np.newaxis is an alias for None:
```
# np.newaxis = None

print('buffer[:, np.newaxis]=', buffer[:, np.newaxis])
$buffer[:, np.newaxis]= [[0][1][2]]

print('buffer[:, None]=', buffer[:, None])
$buffer[:, None]= [[0][1][2]]

print('buffer[np.newaxis, :]=', buffer[np.newaxis, :])
$buffer[np.newaxis, :]= [[0 1 2]]

print('buffer[None, :]=', buffer[None, :])
$buffer[None, :]= [[0 1 2]]
```

Stacking:
```
a = [1,2,3]
b = [4,5,6]
c = [7,8,9]

r = np.hstack((a,b,c)) # horizontal stacking
print("r=", r)
$r= [1 2 3 4 5 6 7 8 9]

QUEUE = queue.Queue()
QUEUE.put(a)
QUEUE.put(b)
QUEUE.put(c)

r = [QUEUE.get() for _ in range(QUEUE.qsize())]
print(r)
$[[1, 2, 3], [4, 5, 6], [7, 8, 9]]

r = np.vstack(r) # vertical stacking
print(r)
$[[1 2 3]
  [4 5 6]
  [7 8 9]]

print(r[:, ::-1]) # col reversal
$[[3 2 1]
  [6 5 4]
  [9 8 7]]
```
