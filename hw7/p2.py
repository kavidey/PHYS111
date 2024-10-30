# %% [markdown]
# ### Imports and Utility Functions
# %%
from IPython.display import display
from sympy import symbols, cos, latex, init_printing, print_latex, diff, Symbol, Eq, solve, Matrix, Transpose, Determinant, Subs, simplify, nsimplify, MatMul, MatAdd, re, Pow, E, exp_polar, I, pi
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
# ### System Matrices
# %%
# Constants
g, k, l, M, m = symbols("g, k, l, M, m", real=True)
# Degrees of Freedom
x, theta, y = dynamicsymbols(r"x, \theta, y", real=True)
x_dot = x.diff()
theta_dot = theta.diff()
y_dot = y.diff()

t = Symbol('t')
# %%
Mm = Matrix([
    [M+m, m],
    [m, m]
])
Km = Matrix([
    [k, 0],
    [0, g*m/l]
])
display(pprint_eqn(M=Mm))
display(pprint_eqn(K=Km))
# %%
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

print("Specific Solution with M=1000kg, m=0.1kg, l=1.55cm, k=158 N/m, g=-9.8m/s^2")
display(list(map(lambda sol: sol.subs(((M,1), (m, 0.1), (l, 1.55e-2), (k, 158), (g, 9.81))), omega_sols)))
numerical_solutions = list(map(lambda sol: sol.subs(((M,1000), (m, 0.1), (l, 1.55e-2), (k, 158), (g, 9.81))).evalf(), omega_sols))
numerical_solutions = [float(f) for f in numerical_solutions if f > 0]
print(numerical_solutions)
# %%
eigenvalues = numerical_solutions
eigenvalue_eqn = (Km - omega**2 * Mm)
a, b = symbols("a, b")
Q = Matrix([[a],[b]])
display(Eq(MatMul(eigenvalue_eqn, Q, evaluate=False), 0, evaluate=False))

eqn_to_solve = (eigenvalue_eqn*Q)[0]
sol = solve(eqn_to_solve, a)[0]
eigenvectors = []
for eigenvalue in eigenvalues:
    display(Eq(omega, eigenvalue))
    b_eqn = sol.subs(((M,1), (m, 0.1), (l, 1.55e-2), (k, 158), (g, 9.81), (omega, eigenvalue)))
    display(Eq(a,b_eqn))
    eigenvector = Matrix([[b_eqn.subs(b, 1)],[1]])
    eigenvectors.append(Matrix([[b_eqn.subs(b, 1)],[1]]))
    display(eigenvector)
    print()
# %%
print("Solution is in the form")
A1, d1, omega1, A2, d2, omega2 = symbols(r"A_1, d_1, \omega_1 A_2, d_2, \omega_2")
time_soln = MatAdd(MatMul(eigenvectors[0], A1*cos(omega1*t+d1), evaluate=False), MatMul(eigenvectors[1], A2*cos(omega2*t+d2), evaluate=False))
display(Eq(Matrix([[x],[y]]), time_soln))
time_soln = A1*eigenvectors[0]*cos(omega1*t+d1) + A2*eigenvectors[1]*cos(omega2*t+d2)

pos = time_soln.subs(((omega1, eigenvalues[0]), (omega2, eigenvalues[1])))
vel = pos.diff(t)

display(Eq(Matrix([[x],[y]]), pos))
display(Eq(Matrix([[x_dot],[y_dot]]), vel))
# %%
x_ic = 2e-2
x_dot_ic = 0
y_ic = 0
y_dot_ic = 0

print("### Solve for phase shift ###")
d_sols = solve(
    [
        # Eq(pos[0].subs(t,0), x_ic),
        # Eq(pos[1].subs(t,0), y_ic),
        Eq(vel[0].subs(t,0), x_dot_ic),
        Eq(vel[1].subs(t,0), y_dot_ic)
    ],
    [d1, d2]
)
display(d_sols)
# %%
print("### Solve for amplitude ###")
for s in d_sols:
    sols = ((d1, s[0]), (d2, s[1]))

    print("Solving with phase shift as")
    display(sols)
    display(solve(
        [
            Eq(pos[0].subs(t,0).subs(sols), x_ic),
            Eq(pos[1].subs(t,0).subs(sols), y_ic),
            Eq(vel[0].subs(t,0).subs(sols), x_dot_ic),
            Eq(vel[1].subs(t,0).subs(sols), y_dot_ic)
        ],
        [A1, A2]
    ))
# %%
C1, C2 = symbols("C_1, C_2", complex=True)
time_soln = C1*eigenvectors[0]*exp_polar(I*omega1*t) + C2*eigenvectors[1]*exp_polar(I*omega2*t)
display(time_soln)

pos = time_soln.subs(((omega1, eigenvalues[0]), (omega2, eigenvalues[1])))
vel = pos.diff(t)

solve([pos[0].subs(t,0)-x_ic, pos[1].subs(t,0)-y_ic], [A1, A2])
# %%
