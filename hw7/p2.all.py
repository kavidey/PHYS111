# %% [markdown]
# ### Imports and Utility Functions
# %%
from IPython.display import display
from sympy import symbols, cos, latex, init_printing, print_latex, diff, Symbol, Eq, solve, Matrix, Transpose, Determinant, Subs, simplify, nsimplify
from sympy.physics.vector import dynamicsymbols
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

latexReplaceRules = {
    r'{\left(t \right)}':r' ',
    r'\frac{d}{d t}':r'\dot',
    r'\frac{d^{2}}{d t^{2}}':r'\ddot',
}
def latexNew(expr,**kwargs):
    retStr = latex(expr,**kwargs)
    for _,__ in latexReplaceRules.items():
        retStr = retStr.replace(_,__)
    return retStr
init_printing(latex_printer=latexNew)
#  %%
def get_solution(diffeq, initial_conditions, tstart=None, tend=None, pts=1000, t=None, **kwargs):
    if t is None:
        t = np.linspace(tstart, tend, int(pts))

    # return odeint(diffeq, initial_conditions, t, args=params, **kwargs).T
    return solve_ivp(diffeq, (np.min(t), np.max(t)), initial_conditions, t_eval=t, **kwargs)

def pprint_eqn(**kwargs):
    name = list(kwargs.keys())[0]
    return Eq(Symbol(name), kwargs[name], evaluate=False)
# %% [markdown]
# ### System Lagrangian
# %%
# Constants
g, k, l, M, m = symbols("g, k, l, M, m", real=True)
# Degrees of Freedom
x, theta, y = dynamicsymbols(r"x, \theta, y", real=True)
x_dot = x.diff()
theta_dot = theta.diff()
y_dot = y.diff()

t = Symbol('T')
# %%
# Potential Energy
U = -m*g*l*cos(theta) + (1/2)*k*x**2
display(pprint_eqn(U=U))
print("substitute y=theta*l")
U = -m*g*l*cos(y/l) + (1/2)*k*x**2
display(pprint_eqn(U=U))
# %%
# Kinetic Energy
T = (1/2)*M*x_dot**2 + (1/2)*m*(x_dot**2 + theta_dot**2 + 2*x_dot*l*theta_dot*cos(theta))
display(pprint_eqn(T=T))
print("substitute y=theta*l")
T = (1/2)*M*x_dot**2 + (1/2)*m*(x_dot**2 + y_dot**2 + 2*x_dot*y_dot*cos(y/l))
display(pprint_eqn(T=T))
# %%
# Lagrangian
L = T-U
pprint_eqn(L=L)
# %% [markdown]
# ### Equilibrium Points
# %%
# Equilibrium Points
print("X Equilibrium")
x_equilibrium = Eq(diff(U, x), 0)
display(x_equilibrium)
x_0 = solve(x_equilibrium, x)[0]
display(pprint_eqn(x=x_0))

print()
print("y Equilibrium")
y_equilibrium = Eq(diff(U, y), 0)
display(y_equilibrium)
y_0 = solve(y_equilibrium, y)[0]
display(pprint_eqn(y=y_0))

# %% [markdown]
# ### Expand L around x0 and y0 and calculate K and M matricies
# %%
# Approximate Potential Energy
U_approx = U.subs(x, x_0).subs(y, y_0) + \
(1/2) * U.diff(x,x).subs(x, x_0).subs(y, y_0) * x**2 + \
(1/2) * U.diff(y, y).subs(x, x_0).subs(y, y_0) * y**2 + \
(1/2) * U.diff(x, y).subs(x, x_0).subs(y, y_0) * x * y

display(pprint_eqn(U=U_approx))

Km = nsimplify(Matrix([
    [U_approx.diff(x,x), U_approx.diff(x,y)],
    [U_approx.diff(x,y), U_approx.diff(y, y)]
]))
display(pprint_eqn(K=Km))
# %%
# Approximate Kinetic Energy
T_approx = (1/2) * T.diff(x_dot, x_dot).subs(x, x_0).subs(y, y_0) * x_dot**2 + \
(1/2) * T.diff(y_dot, y_dot).subs(x, x_0).subs(y, y_0) * y_dot**2 + \
T.diff(x_dot, y_dot).subs(x, x_0).subs(y, y_0) * x_dot*y_dot

display(pprint_eqn(T=T_approx))

Mm = nsimplify(Matrix([
    [T_approx.diff(x_dot,x_dot), T_approx.diff(x_dot,y_dot)],
    [T_approx.diff(x_dot,y_dot), T_approx.diff(y_dot, y_dot)]
]))
display(pprint_eqn(M=Mm))
# %% [markdown]
# ### Matrix form of Lagrangian
# $$L=\frac{1}{2}X^T M X - \frac{1}{2} X^T K X$$
# The matrix Euler-Lagrange Equation is then:
# $$M \ddot X = -K X \implies M \ddot Z = - K Z$$
# Finally we can solve for the normal modes by evaluating
# $$(K-\omega^2 M) Z_0 = 0 \implies \det \left( K-\omega^2 M \right) = 0$$
# %%q
omega = symbols(r"\omega")
inner_matrix = Km- omega**2 * Mm
det_eq = inner_matrix[0]*inner_matrix[3] - inner_matrix[1]*inner_matrix[2]
display(Eq(Determinant(inner_matrix), det_eq, evaluate=False))
# %%
print("General Solution")
omega_sols = solve(
    det_eq,
    omega
)
display(omega_sols)

print("Specific Solution with M=1kg, m=0.1kg, l=1.55cm, k=158 N/m, g=-9.8m/s^2")
display(list(map(lambda sol: simplify(sol.subs(((M,1), (m, 0.1), (l, 1.55e-2), (k, 158), (g, 9.81)))), omega_sols)))
# %%
