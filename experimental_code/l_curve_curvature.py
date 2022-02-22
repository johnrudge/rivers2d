#!/usr/bin/env python
# An experimental code for calculating both the L-curve and its curvature

import sys
sys.path.append("..")

from rivers_2d_assembly import *     # main matrix assembly routines
from rivers_output import *          # output routines
from numpy import array, arange, append, reshape, log10, linspace # basic matrix algebra
from scipy.integrate import cumtrapz # integration
from dolfin import refine            # mesh refinement

# Input data & output directories
input_dir = "../madagascar_data/"

# Mesh file
mesh_file = input_dir + "madagascar.xml"

# River model parameters
v = 200.0
m = 0.2

# Smoothing parameters
lambda_time = 0.0    # time smooth
# set range of space smoothing to try
n_lambda_space = 40
log_lambda_space_min = -2.0
log_lambda_space_max = +2.0
curvature_fraction = 5.0 # parameter used when calculating nearby points for L-curve curvature estimation

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
print('Times = ', t, "Ma")
print('Number of times = ', nt, '  Number of vertices = ', nv)

# Assemble matrices
[Ms_model, bs_model] = assemble_model(mesh, t, river_data)
[S_space, b_space] = assemble_space_smooth(mesh, t)
[S_time, b_time] = assemble_time_smooth(mesh, t)

# Combine matrices into a single list
Ms = Ms_model + [S_space, S_time]
bs = bs_model + [b_space, b_time]

# Perform least squares inversion, weighting terms appropriately
river_weights = [1.0] * nriver  # weight all rivers the same
pins = array([False]*nt*nv) # by default don't pin any values
# pins[(nt-1)*nv:nt*nv] = True # pin the values for the last time step

# Lambda space values to try
log_ls = linspace(log_lambda_space_min, log_lambda_space_max, n_lambda_space)
ls = pow(10,log_ls)

def calc_curvature(x1, y1, x2, y2, x3, y3):
    """ Calculate the curvature gievn three points at (x1, y1), (x2, y2) and (x3, y3)"""
    numerator = 2.0 * (x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2)
    den_1 = (x2-x3)**2 + (y2-y3)**2
    den_2 = (x3-x1)**2 + (y3-y1)**2
    den_3 = (x1-x2)**2 + (y1-y2)**2
    denominator = sqrt(den_1*den_2*den_3)
    curvature = numerator / denominator
    return curvature

model_misfit = zeros(len(ls))
space_misfit = zeros(len(ls))
time_misfit = zeros(len(ls))
l_curve_curvature = zeros(len(ls))

for i, lambda_space in enumerate(ls):
    lambda_space_plus = curvature_fraction * lambda_space
    lambda_space_minus = lambda_space / curvature_fraction
    
    weights = river_weights + [lambda_space, lambda_time]
    
    weights_plus = river_weights + [lambda_space_plus, lambda_time]
    
    weights_minus = river_weights + [lambda_space_minus, lambda_time]
        
    result, fit, residual, misfit = lsq_inversion(Ms, bs, weights, 'nnls_bfgs', pins)

    result_plus, fit_plus, residual_plus, misfit_plus = lsq_inversion(Ms, bs, weights_plus, 'nnls_bfgs', pins)

    result_minus, fit_minus, residual_minus, misfit_minus = lsq_inversion(Ms, bs, weights_minus, 'nnls_bfgs', pins)

    model_misfit[i] = sum(misfit[:-2])
    time_misfit[i] = misfit[-1]
    space_misfit[i] = misfit[-2]

    model_misfit_plus = sum(misfit_plus[:-2])
    model_misfit_minus = sum(misfit_minus[:-2])
    
    space_misfit_plus = misfit_plus[-2]
    space_misfit_minus = misfit_minus[-2]
    
    l_curve_curvature[i] = calc_curvature(log10(model_misfit[i]),log10(space_misfit[i]),log10(model_misfit_plus),log10(space_misfit_plus),log10(model_misfit_minus),log10(space_misfit_minus))
    
    print(lambda_time, lambda_space, model_misfit[i], space_misfit[i], time_misfit[i], l_curve_curvature[i])

xv = model_misfit
yv = space_misfit

# Plotting
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(ls, l_curve_curvature)
plt.xscale('log')
plt.xlabel('lambda_space')
plt.ylabel('L-curve curvature')

plt.figure(2)
labels = ['lam={0}'.format(l) for l in ls]
plt.plot(xv, yv,'ko-')
for label, x, y in zip(labels, xv, yv):
    plt.annotate(label,xy = (x, y), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
plt.xscale('log')
plt.yscale('log')
plt.title("L-curve")
plt.xlabel("Model misfit")
plt.ylabel("Space smoothing misfit")

plt.show()
