# %%
import matplotlib.pyplot as plt
import numpy as np
# %%
M = 1000
m = 100
m0 = 10

r_M = np.array([-2, 1])
r_m = np.array([1, 0.5])
r_m0 = np.array([1.5, 0.25])


R = (M*r_M + m*r_m + m0*r_m0)/(M+m+m0)
r
# %%
def plot_objects(M, R):
    r = np.vstack(R)
    m = np.hstack(M)
    plt.scatter(r[:,0], r[:,1], s=np.log10(m)*40)
# %%
plot_objects((M, m, m0), (r_M, r_m, r_m0))
plt.grid()
plt.xlim(-5,5)
plt.ylim(-5,5)

plt.scatter([R[0]], [R[1]], c='green')
# %%
