---
layout: posts
author: Huan
title: Output dimension from convolution layer
---

How to calculate dimension of output from a convolution layer?

---

#No padding (aka valid padding):

input  = n x n = 6 x 6

kernel = f x f = 3 x 3

output = m x m = 4 x 4

How do we get output = 4 x 4 ?

Ans: Use the formula: (n - f + 1) x (n - f + 1)

---

#With padding of size 1:

p = 1

input  = n x n = 6 x 6

kernel = f x f = 3 x 3

output = m x m = 6 x 6

How do we get output = 6 x 6 ?

Ans: Use the formula: (n + 2p - f + 1) x (n + 2p - f + 1)

---

#Meaning of valid padding & same padding:

1) No padding is also known as valid padding.

2) Same padding means pad input so that the resulting output dimension after
convolution will be the same as input.

---

#Size of padding needed to achieve same padding:

Size of padding needed to achieve same padding depends on the kernel size, f.

Using p = (f - 1) / 2 will produce output dimension = input dimension.

---
<br>
