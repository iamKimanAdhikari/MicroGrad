## 🎓 Motivation & Acknowledgments

This project was built as a deep-dive into the "first principles" of Deep Learning. The implementation was heavily inspired by **Andrej Karpathy’s** "The spelled-out intro to neural networks and backpropagation: building micrograd."

The goal was to move beyond high-level APIs like PyTorch or TensorFlow and manually implement the mathematical machinery that powers modern AI.

---

## 🧠 Concepts Implemented

Building this engine required a hands-on understanding of several core CS and Math concepts:

* **Reverse-Mode Autodiff:** Implementing the chain rule across a dynamically constructed graph.
* **The Chain Rule:** Manually defining local derivatives for operations like `tanh`, `pow`, and `mul`.
* **Topological Sorting:** Using a Depth-First Search (DFS) to ensure that in a directed acyclic graph (DAG), every parent node's gradient is computed before its children.
* **Gradient Accumulation:** Solving the "multivariate chain rule" problem where a variable used multiple times must have its gradients summed rather than overwritten.
* **Symmetry Breaking:** Understanding why weight initialization (randomness) is necessary for neurons to learn distinct features.

---

## 🛠️ Key Implementation Detail: The "Backward" Function

One of the most interesting parts of this project is how the `_backward` logic is stored. Instead of calculating gradients immediately, each operation defines a closure:

```python
def __mul__(self, other):
    # ...
    def _backward():
        self.gradient += other.data * out.gradient
        other.gradient += self.data * out.gradient
    out._backward = _backward

```

This allows the engine to wait until the entire forward pass is complete before "unrolling" the math in reverse order.

---