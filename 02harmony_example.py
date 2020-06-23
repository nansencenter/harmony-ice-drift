import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

n = np.load('field_20190101T000000Z.npz')
# Example 1
# Plot concentration and velocity using Triangulation
xlim = [400000, 800000]
ylim = [-800000, -400000]
plt.tripcolor(n['x'], n['y'], n['Concentration'], triangles=n['i'], vmin=0, vmax=1)
q = plt.quiver(n['x'], n['y'], n['u'], n['v'], angles='xy', scale=10)
plt.quiverkey(q, X=0.3, Y=1.1, U=1, label='Drift: 1 m/s', labelpos='E')
plt.xlim(xlim)
plt.ylim(ylim)
plt.show()


# Example 2
# Compute average speed for each element
u_elems = n['u'][n['i']]
v_elems = n['v'][n['i']]
u_avg = u_elems.mean(axis=1)
v_avg = v_elems.mean(axis=1)
spd = np.hypot(u_avg, v_avg)

# Example 3
# Rasterize concentration (convert from triangle elements to 2D grid)
# 1. Get x,y coordinates of each element
xe, ye = [n[i][n['i']].mean(axis=1) for i in ['x', 'y']]
# 2. Create x,y destination grids
xg, yg = np.meshgrid(np.linspace(*xlim, 100), np.linspace(*ylim[::-1], 100))
# 3. Interpolate from elements onto grid
cg = griddata(np.array([xe, ye]).T, n['Concentration'], np.array([xg, yg]).T).T
plt.imshow(cg, extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
plt.show()
