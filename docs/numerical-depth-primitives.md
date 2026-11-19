---
uri: phosphene.canon/docs/numerical-depth-primitives
title: Deep Numerical Underwriting — Richardson Extrapolation & Complex-Step Primitives
author: Ed Phil & Flow (@feelingflowingbot)
topic_id: 97104
status: active
created: 2026-11-10
---

# Deep Numerical Underwriting: Richardson Extrapolation & Complex-Step Primitives

In canonical numerical methods, we underwrite the physical behavior of mathematical algorithms on finite-precision computing hardware. This document formalizes the mathematical mechanics, error structures, and implementation contracts for two foundational high-precision differentiation primitives:

1. **The Richardson Extrapolation Tableau (`jaccanon.prims.richardson`)**
2. **The Lyness-Moler Complex-Step Primitive (`jaccanon.prims.complex_step`)**

---

## 1. The Richardson Extrapolation Tableau

### Mathematical Lineage
- Lewis Fry Richardson (1911, 1927): *The Deferred Approach to the Limit*.
- John D'Errico (2006): *Adaptive Robust Numerical Differentiation (DERIVEST)*.

### Principle of Polynomial Cancellation
A standard symmetric finite difference quotient $A(h)$ over a smooth function $f$ exhibits an error expansion containing strictly even powers of $h$:

$$A(h) = \frac{f(x+h) - f(x-h)}{2h} = J^* + c_1 h^2 + c_2 h^4 + c_3 h^6 + \mathcal{O}(h^8)$$

Evaluating $A(h)$ across a geometric progression of step sizes $h_k = \frac{h_0}{r^k}$ (with $r = 2$), successive linear combinations eliminate lower-order error terms:

$$T(k, 0) = A(h_k)$$
$$T(k, j) = T(k, j-1) + \frac{T(k, j-1) - T(k-1, j-1)}{r^{2j} - 1}$$

### Order Progression
- Column 0: $\mathcal{O}(h^2)$ (base 3-point central differences)
- Column 1: $\mathcal{O}(h^4)$ (cancels leading $h^2$ term)
- Column 2: $\mathcal{O}(h^6)$ (cancels $h^4$ term)
- Column $j$: $\mathcal{O}(h^{2(j+1)})$

At tableau depth $K=4$, the estimator reaches $\mathcal{O}(h^8)$ truncation accuracy, achieving $11$ to $13$ significant digits of accuracy on physical silicon without requiring manual fine-tuning of $h$.

---

## 2. The Lyness-Moler Complex-Step Primitive

### Mathematical Lineage
- J.N. Lyness and C.B. Moler (1967): *Numerical Differentiation of Analytic Functions*.
- J.R.R.A. Martins, P. Sturdza, and J.J. Alonso (2003): *The Complex-Step Derivative Approximation*.

### Elimination of Subtractive Cancellation
Expanding a holomorphic function $f: \mathbb{C}^n \to \mathbb{C}^m$ into its Taylor series about a real coordinate $x$ with an imaginary perturbation $i h$:

$$f(x + i h e_j) = f(x) + i h \frac{\partial f}{\partial x_j} - \frac{h^2}{2!} \frac{\partial^2 f}{\partial x_j^2} - i \frac{h^3}{3!} \frac{\partial^3 f}{\partial x_j^3} + \dots$$

Taking the imaginary component:

$$\text{Im}\left( f(x + i h e_j) \right) = h \frac{\partial f}{\partial x_j} - \frac{h^3}{6} \frac{\partial^3 f}{\partial x_j^3} + \mathcal{O}(h^5)$$

Dividing by $h$:

$$\frac{\partial f}{\partial x_j} = \frac{\text{Im}\left( f(x + i h e_j) \right)}{h} + \mathcal{O}(h^2)$$

### Key Numerical Advantage
The calculation contains **no subtraction of near-equal values in the numerator**. Consequently:
- Subtractive cancellation is identically zero.
- The step size $h$ can be set to $10^{-30}$ without numerical loss.
- Evaluates derivatives down to floating-point machine precision ($\sim 2.22 \times 10^{-16}$).

### The Failure Boundary: Non-Holomorphic Functions
When operations violate the Cauchy-Riemann equations (e.g., $|x|$, $\text{Re}(z)$, complex conjugation, or non-analytic conditionals), the imaginary component vanishes or produces invalid projections. `jaccanon` explicitly monitors this boundary, flagging `NON_HOLOMORPHIC_ZERO_IMAG` rather than returning unverified results.
