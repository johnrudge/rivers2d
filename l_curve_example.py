#!/usr/bin/env python

from rivers_2d_assembly import *     # main matrix assembly routines
from rivers_output import *          # output routines
from numpy import array, arange, append, reshape, log10 # basic matrix algebra
from scipy.integrate import cumtrapz # integration
from dolfin import refine            # mesh refinement

# Input data & output directories
input_dir = "madagascar_data/"
output_dir = "madagascar_output/"

# Mesh file
mesh_file = input_dir + "madagascar.xml"

# River model parameters
v = 200.0
m = 0.2

# Smoothing parameters
lambda_space = 1.0   # space smooth
lambda_time = 0.0    # time smooth

# Numerical parameters
dt = 2.0            # time spacing (Myr)
n_refinements = 0   # number of times to perform mesh refinement

# Read in preformatted mesh and refine given number of times
mesh = Mesh(mesh_file)
for k in range(n_refinements):
    mesh = refine(mesh)
nv = mesh.num_vertices()

# Read in river data and calculate Gilbert times
river_data = parse_river_data(input_dir, "obs_river*", v, m)
nriver = len(river_data)

# Find max time
max_time = 0.0
for river_dict in river_data:
    rt = river_dict['t']
    max_time = max(max_time, rt[-1])

# Work out time blocks
tint = arange(int(max_time/dt)+1)*dt
t = append(tint,max_time)
nt = len(t)
print 'Times = ', t, "Ma"
print 'Number of times = ', nt, '  Number of vertices = ', nv

# Assemble matrices
[Ms_model, bs_model] = assemble_model(mesh, t, river_data)
[S_space, b_space] = assemble_space_smooth(mesh, t)
[S_time, b_time] = assemble_time_smooth(mesh, t)

# Combine matrices into a single list
Ms = Ms_model + [S_space, S_time]
bs = bs_model + [b_space, b_time]

# Perform least squares inversion, weighting terms appropriately
river_weights = [1.0] * nriver  # weight all rivers the same
weights = river_weights + [lambda_space, lambda_time]
pins = array([False]*nt*nv) # by default don't pin any values
# pins[(nt-1)*nv:nt*nv] = True # pin the values for the last time step


ls = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
model_misfit = zeros(len(ls))
space_misfit = zeros(len(ls))
time_misfit = zeros(len(ls))
for i, lambda_space in enumerate(ls):
#for i, lambda_time in enumerate(ls):
    weights = river_weights + [lambda_space, lambda_time]
    result, fit, residual, misfit = lsq_inversion(Ms, bs, weights, 'nnls_bfgs', pins)
    model_misfit[i] = sum(misfit[:-2])
    time_misfit[i] = misfit[-1]
    space_misfit[i] = misfit[-2]
    
    print lambda_time, lambda_space, model_misfit[i], space_misfit[i], time_misfit[i]

xv = model_misfit
yv = space_misfit

# Plotting
import matplotlib.pyplot as plt

labels = ['lam={0}'.format(l) for l in ls]
plt.plot(log10(xv), log10(yv),'ko-')
for label, x, y in zip(labels, xv, yv):
    plt.annotate(label,xy = (log10(x), log10(y)), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
plt.title("L-curve")
plt.xlabel("Model misfit")
plt.ylabel("Space smoothing misfit")
plt.show()
