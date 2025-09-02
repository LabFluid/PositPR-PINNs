# PositPR-PINNs-code

Julia codes for experiments with Posit arithmetic in Physics-Informed Neural Networks (PINNs).  
Requires the custom PositPR.jl library, currently available only locally.

## Differential Equations used in the experiments

### eq1 (ode)

**Differential equation**

$$
\frac{du}{dt} = \cos(2\pi t)
$$

**Condition**

$$
u(0) = 0
$$

**Analytical solution**

$$
u(t) = \frac{\sin(2\pi t)}{2\pi}
$$

---

### eq2 (pde)

**Differential equation**

$$
\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}
$$

**Conditions**

$$
u(0,t) = u(1,t) = 0, \qquad u(x,0) = \sin(\pi x)
$$

**Analytical solution**

$$
u(x,t) = \sin(\pi x)\, e^{-\pi^2 t}
$$

