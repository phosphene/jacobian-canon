---
title: Automated Routing: Conservative vs. Non-Conservative Vector Fields
author: Ed Phil & Flow (@feelingflowingbot)
status: living
created: 2026-11-10
---

# Automated Routing: Conservative vs. Non-Conservative Vector Fields

A central contribution of `jaccanon` is bridging the gap between local Jacobian differentiation and global dynamical stability.

When analyzing or solving an equilibrium for a vector field $f: \mathbb{R}^n \to \mathbb{R}^n$, the Jacobian $J = \nabla f(x)$ determines the local flow structure. Standard optimization frameworks apply gradient descent or Newton-type schemes. In vector fields characterized by non-conservative rotational curl, gradient updates oscillate across periodic limit cycles or diverge into explosive orbits.

---

## 1. The Helmholtz Matrix Splitting

Any square matrix $M \in \mathbb{R}^{n \times n}$ decomposes uniquely into symmetric and antisymmetric components:

$$M = S + A$$

Where:
- **$S = \frac{1}{2}(M + M^T)$:** The **symmetric operator**, representing the potential gradient (irrotational diffusion or monotone dissipation).
- **$A = \frac{1}{2}(M - M^T)$:** The **antisymmetric operator**, representing non-conservative rotational torque (curl, borrowing dynamics, or zero-sum game dynamics).

The two components are strictly orthogonal in the Frobenius inner product:
$$\langle S, A \rangle_F = \text{tr}(S^T A) = \sum_{i,j} S_{ij} A_{ij} = 0$$

---

## 2. The Symmetrization Threshold Metric ($\rho$)

To classify the dynamical regime, we define the dimensionless **asymmetry ratio**:

$$\rho = \frac{\|A\|_F}{\|S\|_F}$$

Given a configured threshold $\epsilon_{\text{sym}}$ (default $0.05$):

### Regime 1: Conservative ($\rho < \epsilon_{\text{sym}}$)
- The vector field is dominated by potential gradients.
- Eigenvalues of $J$ are predominantly real: $\text{Re}(\lambda) \gg |\text{Im}(\lambda)|$.
- Single-step monotone gradient descent converges reliably.
- **Automated Routing:** `gradient_descent`.

### Regime 2: Non-Conservative ($\rho \ge \epsilon_{\text{sym}}$)
- Rotational modes dominate the dynamics.
- Imaginary components of the spectrum induce cycling and orbital drift.
- Standard gradient descent overshoots and diverges on pure rotations.
- **Automated Routing:** `extragradient_projection` (Korpelevich two-step predictor-corrector).

---

## 3. The Korpelevich Extragradient Solver

For non-conservative vector fields, the Korpelevich algorithm applies a two-step prediction-correction step that dampens rotational cycles:

1. **Prediction Step (Lookahead):**
   $$\bar{z}^{(k)} = \Pi_{\Omega} \left( z^{(k)} - \gamma f(z^{(k)}) \right)$$
2. **Correction Step (Evaluation at Predicted State):**
   $$z^{(k+1)} = \Pi_{\Omega} \left( z^{(k)} - \gamma f(\bar{z}^{(k)}) \right)$$

Where $\Pi_{\Omega}$ denotes Euclidean projection onto the feasible set $\Omega$ (e.g., the probability simplex via Condat's projection, or box constraints).

By evaluating the vector field at the intermediate predicted state $\bar{z}^{(k)}$, the method generates a second-order contraction against the rotational curl, stabilizing limit cycles that destroy standard descent algorithms.
