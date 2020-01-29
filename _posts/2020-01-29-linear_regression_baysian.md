---
layout: posts
author: Huan
title: Linear regression (Baysian)
---

Linear regression (Baysian)

---

Bayes' theorem:

$$P(a, b) = P(a|b) P(b) = P(b|a) P(a) -> P(a|b) = P(b|a) P(a) / P(b)$$

So,

$$P(y, w|x) = P(x|y, w) P(y, w) / P(x) ---> 1$$

The 1st part of the nominator from 1 is:

$$P(x|y, w) ---> 2$$

From joint probability:

$$P(a, b, c) = P(a|b, c) P(b, c) = P(b|a, c) P (a, c)$$

$$i.e. P(a|b, c) = P(b|a, c) P(a, c) / P(b, c) ---> 3$$

Apply 3 to 2:

$$P(x|y, w) = P(y|x, w) P(x, w) / P(y, w) ---> 4$$

Plug 4 back to 1:

$$P(y, w|x) = [ P(y|x, w) P(x, w) P(y, w) ] / [P(y, w) P( x)]$$

$$P(y, w|x) = [P(y|x, w) P(x, w)] / P(x)$$

$$P(y, w|x) = [P(y|x, w) P(w|x) P(x)] / P(x)$$

$$P(y, w|x) = P(y|x, w) P(w|x) ---> 5$$

If w and x are independent:

$$P(y, w|x) = P(y|x, w) P(w) ---> 6$$

Also from 5, if we switch w with y, we can obtain:

$$P(w, y|x) = P(w|x, y) P(y|x)$$

$$P(y, w|x) = P(w|x, y) P(y|x)$$

$$P(w|x, y) = P(y, w|x) / P(y|x) ---> 7$$

We try to maximize 7 with respect to w. Only the nominator depends on w, so we
can ignore the denominator P(y|x) and we get:

Maximize with respect to w in the following equations 8, 9, 10:

$$P(w|x, y) = P(y, w|x) ---> 8$$

Therefore, from 6 & 8 we get:

$$P(w|x, y) = P(y, w|x) = P(y|x, w) P(w) ---> 9$$

$$P(w|x, y) = P(y|x, w) P(w) ---> 10$$

---

<br>
