---
layout: posts
author: Huan
title: Quantum States as a generalization of classical probabilities.
---

Notes on quantum states as a generalization of classical probabilities.

---

# Assumption:

Assume (1 / sqrt(2)) is factored out (not considered in the below examples).

---

# |0> + |1>

```
ket_0 = np.array([[1],
                  [0]])

ket_1 = np.array([[0],
                  [1]])

hadamard = np.array([[1, 1],
                     [1, -1]])

print("ket_0 + ket_1 =", ket_0 + ket_1)
print("hadamard @ ket_0 =", hadamard @ ket_0)

print("ket_0 - ket_1 =", ket_0 - ket_1)
print("hadamard @ ket_1 =", hadamard @ ket_1)
```

## Output:

```
ket_0 + ket_1 = [[1]
                 [1]]

hadamard @ ket_0 = [[1]
                    [1]]

ket_0 - ket_1 = [[ 1]
                 [-1]]

hadamard @ ket_1 = [[ 1]
                    [-1]]
```

---

# Kronecker product for composite states.

```
ket_00 = np.kron(ket_0, ket_0)
ket_01 = np.kron(ket_0, ket_1)
ket_10 = np.kron(ket_1, ket_0)
ket_11 = np.kron(ket_1, ket_1)

print("|00⟩ : ", ket_00)
print("|01⟩ : ", ket_01)
print("|10⟩ : ", ket_10)
print("|11⟩ : ", ket_11)

print("|00⟩ + |11⟩ =", ket_00 + ket_11)
print("|00⟩ - |11⟩ =", ket_00 - ket_11)
```

## Output:

```
|00⟩ :  [[1]
         [0]
         [0]
         [0]]

|01⟩ :  [[0]
         [1]
         [0]
         [0]]

|10⟩ :  [[0]
         [0]
         [1]
         [0]]

|11⟩ :  [[0]
         [0]
         [0]
         [1]]

|00⟩ + |11⟩ = [[1]
               [0]
               [0]
               [1]]

|00⟩ - |11⟩ = [[ 1]
               [ 0]
               [ 0]
               [-1]]
```

---

# |00> + |11>

```
# Hadamard gate
hadamard = np.array([[1, 1],
                     [1, -1]])

# CNOT gate
CNOT_mat = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])

# Initialize 2 qubits, q_0, q_1 as |0>.
q_0 = np.array([[1],
                [0]])
q_1 = np.array([[1],
                [0]])

# Apply hadamard to q_0.
H_q_0 = hadamard @ q_0
print("H_q_0: {}".format(H_q_0))

# Composite state of q_0 (after applying hadamard to q_0) & q_1.
composite = np.kron(H_q_0, q_1)
print("composite: {}".format(composite))

# Apply CNOT to composite state.
res = CNOT_mat @ composite
print("res: {}".format(res))
```

## Output:

```
H_q_0: [[1]
        [1]]

composite: [[1]
            [0]
            [1]
            [0]]

res: [[1]
      [0]
      [0]
      [1]]
```

---

# |00> - |11>

```
# NOT gate
def Not_gate(v):
    if v[0] == 0:
        v[0] = 1
    else:
        v[0] = 0

    if v[1] == 0:
        v[1] = 1
    else:
        v[1] = 0

    return v

# Hadamard gate
hadamard = np.array([[1, 1],
                     [1, -1]])

# CNOT gate
CNOT_mat = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])

# Initialize 2 qubits, q_0, q_1 as |0>.
q_0 = np.array([[1],
                [0]])
q_1 = np.array([[1],
                [0]])

# Apply NOT to q_1.
X_q_1 = Not_gate(q_1)
print("X_q_1: {}".format(X_q_1))

# Apply hadamard to q_1.
H_q_1 = hadamard @ X_q_1
print("H_q_1: {}".format(H_q_1))

# Composite state of q_1 (after applying hadamard to q_1) & q_0.
composite = np.kron(H_q_1, q_0)   # Need to CNOT(1,0) instead of CNOT(0,1) so q_1 is 1st term in composite.
print("composite: {}".format(composite))

# Apply CNOT to composite state.
res = CNOT_mat @ composite
print("res: {}".format(res))
```

## Output:

```
X_q_1: [[0]
        [1]]

H_q_1: [[ 1]
        [-1]]

composite: [[ 1]
            [ 0]
            [-1]
            [ 0]]

res: [[ 1]
      [ 0]
      [ 0]
      [-1]]
```
---

<br>
