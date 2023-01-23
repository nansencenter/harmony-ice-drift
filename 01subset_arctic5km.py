import os
import glob
import time
import datetime as dt
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np

from pynextsim import NextsimBin

def read_file_save(ifile):
    ofile = ifile.replace('.bin', '.npz')
    if os.path.exists(ofile):
        return
    print(ifile, ofile)
    nvars = {}
    n = NextsimBin(ifile)

    nvars['x'] = n.mesh_info.nodes_x
    nvars['y'] = n.mesh_info.nodes_y
    nvars['lon'], nvars['lat'] = n.mesh_info.projection.pyproj(nvars['x'], nvars['y'], inverse=True)
    nvars['i'] = n.mesh_info.indices
    mvt = n.get_var('M_VT')
    num_nodes = nvars['x'].size
    nvars['u'], nvars['v'] = mvt[:num_nodes], mvt[num_nodes:]

    for varname in ['Concentration', 'Thickness']:
        nvars[varname] = n.get_var(varname)

    np.savez(ofile, **nvars)

def read_file_save_velocity(ifile1, timedelta=dt.timedelta(1)):
    print(ifile1)
    n1 = NextsimBin(ifile1)
    date1 = n1.datetime
    date2 = n1.datetime + timedelta
    ifile2 = os.path.join(
        os.path.split(ifile1)[0],
        'field_%sZ.bin' % date2.strftime('%Y%m%dT%H%M%S'))
    n2 = NextsimBin(ifile2)

    x1 = n1.mesh_info.nodes_x
    t1 = n1.mesh_info.nodes_t
    i1 = n1.mesh_info.get_var('id')

    x2 = n2.mesh_info.nodes_x
    t2 = n2.mesh_info.nodes_t
    i2 = n2.mesh_info.get_var('id')

    # indices of nodes common to 0 and 1
    ids_cmn_12, ids1i, ids2i = np.intersect1d(i1, i2, return_indices=True)

    # coordinates of nodes of common elements
    x1n = x1[ids1i]
    y1n = y1[ids1i]
    x2n = x2[ids2i]
    y2n = y2[ids2i]

    u = (x2n - x1n) / timedelta.total_seconds()
    v = (y2n - y1n) / timedelta.total_seconds()

    tri = Triangulation(x1n, y1n)

    nvars = dict(
        x=x1n,
        y=y1n,
        i=tri.triangles,
        u=u,
        v=v)
    nvars['lon'], nvars['lat'] = n1.mesh_info.mapping(nvars['x'], nvars['y'], inverse=True)
    ofile = ifile1.replace('.bin', '_vel.npz')
    np.savez(ofile, **nvars)


idir = '/data2/antonk/harmony/sa05free*'

ifiles = sorted(glob.glob(f'{idir}/field_*.bin'))

print(len(ifiles), ifiles[0], ifiles[-1])

#read_file_save(ifiles[0])

with Pool(5) as p:
    p.map(read_file_save, ifiles)
#    p.map(read_file_save_velocity, ifiles)

# subset
#ex = n['x'][n['i']]
#ey = n['y'][n['i']]
#crit = (ex > xlim[0]) * (ex <= xlim[1]) * (ey > ylim[0]) * (ey <= ylim[1])
#gpi = np.where(gpi.all(axis=1))[0]
#ni1 = n['i'][gpi]
# TODO: reduce number of nodes in nx, ny; shift element node indices in ni1 correspondingly

#

raise
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

n = np.load(os.path.join(odir, 'field_20190101T000000Z.npz'))
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

# Example 4
# Plot instantaneous and average speed and deformation
n1 = np.load(os.path.join(odir, 'field_20190101T000000Z.npz'))
n2 = np.load(os.path.join(odir, 'field_20190101T000000Z_vel.npz'))
xlim = [400000, 600000]
ylim = [-800000, -600000]

fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,5))
for i, n in enumerate([n1, n2]):
    e1, e2, e3, a, p, t = get_deformation_nodes(n['x'], n['y'], n['u'], n['v'])
    ax[i].tripcolor(n['x'], n['y'], e3*24*60*60, triangles=t, vmin=0, vmax=2, cmap='plasma_r')
    q = ax[i].quiver(n['x'], n['y'], n['u'], n['v'], angles='xy', scale=10)
    ax[i].quiverkey(q, X=0.3, Y=1.1, U=1, label='Drift: 1 m/s', labelpos='E')
    ax[i].set_xlim(xlim)
    ax[i].set_ylim(ylim)
plt.savefig('velocity_and_deformation.png', dpi=300, pad_inches=0, bbox_inches='tight')
plt.close()
