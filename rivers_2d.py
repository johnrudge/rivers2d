#!/usr/bin/env python

from rivers_2d_assembly import *     # main matrix assembly routines
from rivers_output import *          # output routines
from numpy import array, arange, append, reshape, ones, concatenate # basic matrix algebra
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
n_refinements = 1   # number of times to perform mesh refinement

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
result, fit, residual, misfit = lsq_inversion(Ms, bs, weights, 'nnls_bfgs', pins)

# Calculate nullity
nullity = calculate_nullity(Ms_model)

# Reshape calculated uplift to a matrix for ease of use 
newshape = (nt,nv)
U = result.reshape(newshape)
null = nullity.reshape(newshape)
z = cumtrapz(U[::-1,:], x=-t[::-1,None], axis=0)[::-1]
z = concatenate([z, zeros((1,nv))], axis=0)

# Output in different formats for visualisation etc.
print "Writing output"
vtk_dir = output_dir + "vtk/"
ascii_dir = output_dir + "ascii/"
gnuplot_dir = output_dir + "gnuplot/"

# Mesh
output_mesh_ascii(mesh, ascii_dir + "mesh")

# River data
output_river_data_vtk(river_data, vtk_dir + "river_data")

# Fit to rivers
output_fits_ascii(river_data, fit, ascii_dir + "fits/")

# Uplift
output_vector_gnuplot(mesh, t, U, gnuplot_dir + "gp_out")
output_vector_ascii(mesh, t, U,  ascii_dir + "uplift.txt")
output_vector_vtk(mesh, t, U, vtk_dir + "u.pvd")

# Cumulative uplift
output_vector_gnuplot(mesh, t, z, gnuplot_dir + "gp_cum")
output_vector_ascii(mesh, t, z, ascii_dir + "cumulative.txt")
output_vector_vtk(mesh, t, z, vtk_dir + "z.pvd")

# Info map
output_vector_ascii(mesh, t, null,  ascii_dir + "null.txt")
output_vector_vtk(mesh, t, null, vtk_dir + "null.pvd")

# Calculate RMS model misfit (xi scaled)
n_observations=sum([len(b) for b in bs_model])
sigma_z = 20.0 # elevation error
xi_rms_model_misfit = sqrt(sum(misfit[:-2])/n_observations)/sigma_z
print "Xi RMS model misfit = ", xi_rms_model_misfit
