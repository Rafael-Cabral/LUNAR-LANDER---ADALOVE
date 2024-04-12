# Deep Q-Network (DQN) from Scratch — TensorFlow & OpenAI Gym

End-to-end implementation of **Deep Reinforcement Learning (Deep Q-Network / DQN)** in **Python**, **TensorFlow / Keras**, and **OpenAI Gym** (`LunarLander-v2`), featuring a custom `@tf.function` training loop with `tf.GradientTape`, Experience Replay, and Polyak soft target-network updates.

> **Engineering context:** This implementation served as the foundational DQN architecture and training loop later adapted to our financial R&D project with **BTG Pactual**, where we modeled a custom `CashAndCarryArbitrageEnv` to optimize daily matching across 565k+ spot and forward operations toward 100% CDI.

---

## Architecture & RL Formulation

- **Environment (`LunarLander-v2`):** Continuous 8-dimensional state vector $(x, y, v_x, v_y, \theta, \omega, c_L, c_R) \in \mathbb{R}^8$ and discrete action space $\mathcal{A} \in \{0, 1, 2, 3\}$ (no-op, right thruster, main thruster, left thruster).
- **Q-Network & Target Q-Network:** Twin MLP architectures (`Input(8) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(4, Linear)`) stabilizing non-stationary bootstrap targets.
- **Bellman MSE Loss:**
  $$y_i = r_i + \gamma \max_{a'} \hat{Q}(s_i', a'; \theta^-) \cdot (1 - \text{done}_i), \qquad \mathcal{L}(\theta) = \frac{1}{B}\sum_{i=1}^{B}\big(y_i - Q(s_i, a_i; \theta)\big)^2$$
- **Training Stabilization:**
  - **Experience Replay Buffer:** `deque(maxlen=10_000)` with uniform random minibatch sampling ($B = 64$) to decorrelate sequential trajectories.
  - **Compiled Gradient Step:** `@tf.function` graph-compiled `tf.GradientTape` step with `Adam(lr=1e-3)` updated every 3 environment steps.
  - **Polyak Soft Updates:** Target weights updated smoothly via $\theta^- \leftarrow \tau \theta + (1 - \tau)\theta^-$ with $\tau = 10^{-3}$.
  - **Exploration Schedule:** $\varepsilon$-greedy policy decaying exponentially ($0.995$ per episode) from $\varepsilon_0 = 1.0$ down to $\varepsilon_{\min} = 0.01$, with discount factor $\gamma = 0.997$.

---

## Repository Contents

- [`Rafael_Cabral_LunarLander.ipynb`](./Rafael_Cabral_LunarLander.ipynb) — Full notebook with environment inspection, network definition, Bellman loss, `@tf.function` training loop, convergence curves, and MP4 policy rollout generation.
- [Trained Weights & Rollout Video (Google Drive)](https://drive.google.com/drive/folders/1F53y9D9ZyTtJ7wcXmmMV7xZTtWej1IJb?usp=sharing) — Saved model checkpoint and rendered landing episode.
