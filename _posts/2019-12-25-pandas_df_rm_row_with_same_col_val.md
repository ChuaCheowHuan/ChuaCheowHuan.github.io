---
layout: posts
author: Huan
title: Filter rows with same column values in Pandas dataframe
---

Filter rows with same column values in a Pandas dataframe.

---

Assume the following Pandas dataframe where Q1 & Q2 are charges. A charge can
either be +1 or -1.

```
df = pd.DataFrame([[1,1],[-1,-1],[1,-1],[-1,1]], columns=['Q1', 'Q2'])

   Q1  Q2
0   1   1
1  -1  -1
2   1  -1
3  -1   1
```

The following code produces a dataframe where each row only contains
opposite charges, i.e if Q1 is 1, Q2 is -1 & vice versa.

```
df = df[ (df.Q1 == 1) & (df.Q2 == -1) | (df.Q1 == -1) & (df.Q2 == 1) ]

   Q1  Q2
2   1  -1
3  -1   1
```

---

<br>
