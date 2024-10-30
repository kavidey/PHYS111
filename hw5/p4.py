# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import odeint
# %% Part A
x = np.linspace(0, 5, 1000)
f = lambda x: x * np.tan(x)
g = lambda x: np.power(x, 2) / (1+x)

plt.plot(x, f(x), label="f(x)")
plt.plot(x, g(x), label="g(x)")

plt.ylim(-10, 10)

plt.grid()
plt.legend()
plt.show()
# %% Part B
f_to_solve = lambda x: f(x) - g(x) - 1/2
solutions = fsolve(f_to_solve, [0])

plt.plot(x, f_to_solve(x), c='g', label="f(x)-g(x)")

plt.scatter(solutions, np.zeros_like(solutions))

plt.ylim(-10, 10)
plt.grid()
plt.legend()
plt.show()

print(f"Zeros: {solutions}")

# %% Part C
def diffeq(Y, t):
    x, v = Y
    dydt = [v, t**2 - x**2]
    return dydt

x0 = 0
v0 = 0.5
t = np.arange(0, 10, 0.01)
x, v = odeint(diffeq, [x0, v0], t).T

plt.plot(t, x)
plt.xlabel("t")
plt.ylabel("x")

plt.grid()
plt.show()
# %% Part C 2
t = np.arange(-10, 10, 0.01)
x, v = odeint(diffeq, [x0, v0], t).T

plt.plot(t, x)

plt.ylim(-10, 10)
plt.grid()
plt.xlabel("t")
plt.ylabel("x")
plt.show()
# %%
