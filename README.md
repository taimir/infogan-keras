# infogan-keras
---

Implementation of the paper [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) by Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel.

---

The implementation of the model is entirely based on the `keras` APIs. As of now, the model trainer uses `tensorflow` directly to create `TensorBoard` summaries (will be decoupled in the future).

Run with:

```
python main.py <experiment name>
```

Visualize in tensorboard:

```
tensorboard --logdir=<project root>
```
