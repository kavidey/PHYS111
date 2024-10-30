# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve

# %matplotlib widget
# %%
def moving_pendulum(tau, Y, g, L):
    theta, omega = Y
    w0 = np.sqrt(g/L)

    theta_dot = omega
    omega_dot = np.sin(tau) - (g/(L*w0**2)) * np.sin(theta)

    return [theta_dot, omega_dot]

def double_pendulum_theta2(theta2, Y):
    theta1, omega1, omega2 = Y
    theta2 = theta2 % (2*np.pi)
    
    delta_theta = theta2-theta1

    theta1_dot = omega1/omega2

    denominator = omega2 * (1 + np.sin(delta_theta)**2)
    omega1_dot = ((-np.sin(delta_theta) * (omega1**2 * np.cos(delta_theta) + omega2**2)) - (2*np.sin(theta1) - np.sin(theta2) * np.cos(delta_theta))) / denominator
    omega2_dot = (np.sin(delta_theta) * (omega1**2 + omega2**2 * np.cos(delta_theta)) + (np.sin(theta1) * np.cos(delta_theta) - 2*np.sin(theta2))) / denominator

    return [theta1_dot, omega1_dot, omega2_dot]

def double_pendulum_t(t, Y):
    theta1, omega1, theta2, omega2 = Y

    delta_theta = theta2-theta1

    theta1_dot = omega1
    omega1_dot = ((-np.sin(delta_theta) * (omega1**2 * np.cos(delta_theta) + omega2**2)) - (2*np.sin(theta1) - np.sin(theta2) * np.cos(delta_theta))) / (1 + np.sin(delta_theta)**2)
    theta2_dot = omega2
    omega2_dot = (np.sin(delta_theta) * (omega1**2 + omega2**2 * np.cos(delta_theta)) + (np.sin(theta1) * np.cos(delta_theta) - 2*np.sin(theta2))) / (1 + np.sin(delta_theta)**2)

    return [theta1_dot, omega1_dot, theta2_dot, omega2_dot]


def get_solution(diffeq, initial_conditions, tstart=None, tend=None, pts=1000, t=None, **kwargs):
    if t is None:
        t = np.linspace(tstart, tend, int(pts))

    # return odeint(diffeq, initial_conditions, t, args=params, **kwargs).T
    return solve_ivp(diffeq, (np.min(t), np.max(t)), initial_conditions, t_eval=t, **kwargs)
L = 10
g = 9.8
# %% Part A
plt.figure()
initial_conditions = [
    (1,0),
    (1,1),
    (0,0),
    (0,1),
]
for ic in initial_conditions:
    sol = get_solution(lambda tau, Y: moving_pendulum(tau, Y, g, L), ic, tstart=0, tend=20).y
    plt.plot(sol[0], sol[1],  label=f'$\\theta={ic[0]}, \\omega={ic[1]}$')
plt.legend()
plt.grid()
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\omega$')
plt.show()
# %% Part B
plt.figure()
initial_conditions = [
    (1,0),
    (1,1),
    (0,0),
    (0,1),
    (0.1, 0.1)
]

intersections = np.arange(0, int(1e4), 2*np.pi)
for ic in initial_conditions:
    theta, omega = get_solution(lambda tau, Y: moving_pendulum(tau, Y, g, L), ic, t=intersections, method='LSODA').y
    
    r = omega
    plt.scatter(r*np.cos(theta), r*np.sin(theta), alpha=0.5, label=f'$\\theta={ic[0]}, \\omega={ic[1]}$', s=1, zorder=1)
plt.legend()
plt.gca().set_aspect('equal')
plt.grid(zorder=-1)
# plt.show()
# %% Part C: Calculate initial conditions with desired H
hamiltonian = lambda theta1, omega1, theta2, omega2: omega1**2 + (1/2) * omega2**2 + omega1*omega2*np.cos(theta2-theta1) - np.cos(theta1) - np.cos(theta2)
def find_omega1_from_hamiltonian(H, theta1, theta2, omega2):
    # solve for a omega1 that makes the hamiltonian equal to the provided H given the other initial conditions
    hamiltonian_to_solve = lambda omega1: hamiltonian(theta1, omega1, theta2, omega2) - H
    return fsolve(hamiltonian_to_solve, [0])[0]

initial_theta2 = 1
H = 1
print(f"Desired H: {H}")
chosen_ics = [ # H, theta1, theta2, omega2
    # (H, 0, initial_theta2, 6),
    (H, 0, initial_theta2, 5.2),
    (H, 2.1, initial_theta2, 6),
    (H, np.pi/2, initial_theta2, 2*np.pi),
]
initial_conditions = [] # theta1, omega1, omega2
for ic in chosen_ics:
    initial_conditions.append((ic[1], find_omega1_from_hamiltonian(ic[0], ic[1], ic[2], ic[3]), ic[3]))
    print(f"Initial Condition: {initial_conditions}, H: {hamiltonian(initial_conditions[-1][0], initial_conditions[-1][1], initial_theta2, initial_conditions[-1][2]):.3f}")

# %% Part C: Plot solutions in the time domain
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
theta2 = np.linspace(initial_theta2, 100, int(5e5))
# theta2 = np.linspace(initial_theta2, int(8e2), int(1e4))
for ic in initial_conditions:
    theta1, omega1, omega2 = get_solution(double_pendulum_theta2, ic, t=theta2, method='Radau').y
    axs[0].plot(theta1, omega1,  label=f'$\\theta={ic[0]:.2f}, \\omega={ic[1]:.2f}$')
    axs[1].plot(theta2, omega2,  label=f'$\\theta={ic[0]:.2f}, \\omega={ic[1]:.2f}$')
axs[0].legend(loc=9, bbox_to_anchor=(0.5,-0.2))
axs[1].legend(loc=9, bbox_to_anchor=(0.5,-0.2))
axs[0].grid()
axs[1].grid()
axs[0].set_xlabel(r'$\theta_1$')
axs[0].set_ylabel(r'$\omega_1$')
axs[1].set_xlabel(r'$\theta_2$')
axs[1].set_ylabel(r'$\omega_2$')
plt.show()
# %% Part C: Plot surfaces of section
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')

intersections = np.arange(initial_theta2, int(1e4), 2*np.pi)
for ic in initial_conditions:
    theta1, omega1, omega2 = get_solution(double_pendulum_theta2, ic, t=intersections, method='DOP853').y
    
    r = omega1
    theta = theta1
    phi = omega2

    ax.scatter(
        r*np.sin(phi)*np.cos(theta),
        r*np.sin(phi)*np.sin(theta),
        r*np.cos(phi),
        label=f'$\\theta={ic[0]:.2f}, \\omega={ic[1]:.2f}$',
        # alpha=0.1
    )
ax.legend()
ax.set_aspect('equal')
plt.show()
# %%
