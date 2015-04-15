#!/usr/bin/env python

from rivers_2d_assembly import *     # main matrix assembly routines
from rivers_output import *          # output routines
from numpy import array, arange, append, reshape, ones, concatenate, linspace # basic matrix algebra
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
scaled_lambda_space = 0.5   # space smooth
scaled_lambda_time = 0.0    # time smooth

# Numerical parameters
nt = 20            # number of time intervals
n_refinements = 1  # number of times to perform mesh refinement

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
max_elevation = 0.0
max_distance = 0.0
for river_dict in river_data:
    rt = river_dict['t']
    rz = river_dict['z']
    d = river_dict['d']
    
    max_time = max(max_time, rt[-1])
    max_elevation = max(max_elevation, rz[-1])
    max_distance = max(max_distance, d[0])
    
# Determine rescaling 
## Scaling so that longest river has a Gilbert time = 1,
## highest elevation = 1, and dimensionless v = 1
#t_scale = max_time
#d_scale = (v*max_time)**(1.0/(1.0 - 2.0*m))
#x_scale = d_scale / 1000.0
#z_scale = max_elevation

# Scaling so that longest river has a max distance = 1,
# highest elevation = 1, and dimensionless v = 1
t_scale = (1.0/v) * max_distance**(1.0 - 2.0*m)
d_scale = max_distance
x_scale = d_scale / 1000.0
z_scale = max_elevation

print "Time scale", t_scale, "Myr"
print "Horizontal scale", d_scale, "m"
print "Horizontal scale", x_scale, "km"
print "Vertical scale", z_scale, "km"
    
# Rescale input data
scaled_river_data = []
for river_dict in river_data:
    scaled_river_dict = {}
    scaled_river_dict['filename'] = river_dict['filename']
    scaled_river_dict['x'] = river_dict['x']/x_scale
    scaled_river_dict['y'] = river_dict['y']/x_scale
    scaled_river_dict['z'] = river_dict['z']/z_scale
    scaled_river_dict['t'] = river_dict['t']/t_scale
    scaled_river_dict['A'] = river_dict['A']/(d_scale*d_scale)
    scaled_river_dict['d'] = river_dict['d']/d_scale
    scaled_river_data.append(scaled_river_dict)

# Rescale mesh
scaled_mesh = Mesh(mesh)
scaled_mesh.coordinates()[:, :] *= 1.0/x_scale
scaled_mesh.bounding_box_tree().build(scaled_mesh)

# Work out time blocks
scaled_max_time = max_time/t_scale
scaled_t = linspace(0.0, scaled_max_time, nt)

print 'Times = ', scaled_t, "dimensionless time"
print 'Number of times = ', nt, '  Number of vertices = ', nv

# Assemble matrices
[Ms_model, bs_model] = assemble_model(scaled_mesh, scaled_t, scaled_river_data)
[S_space, b_space] = assemble_space_smooth(scaled_mesh, scaled_t)
[S_time, b_time] = assemble_time_smooth(scaled_mesh, scaled_t)

# Combine matrices into a single list
Ms = Ms_model + [S_space, S_time]
bs = bs_model + [b_space, b_time]

# Perform least squares inversion, weighting terms appropriately
river_weights = [1.0] * nriver  # weight all rivers the same
weights = river_weights + [scaled_lambda_space, scaled_lambda_time]
pins = array([False]*nt*nv) # by default don't pin any values
# pins[(nt-1)*nv:nt*nv] = True # pin the values for the last time step
scaled_result, scaled_fit, scaled_residual, scaled_misfit = lsq_inversion(Ms, bs, weights, 'nnls_bfgs', pins)

# Calculate nullity
nullity = calculate_nullity(Ms_model)

# Reshape calculated uplift to a matrix for ease of use 
newshape = (nt,nv)
scaled_U = scaled_result.reshape(newshape)
null = nullity.reshape(newshape)
scaled_z = cumtrapz(scaled_U[::-1,:], x=-scaled_t[::-1,None], axis=0)[::-1]
scaled_z = concatenate([scaled_z, zeros((1,nv))], axis=0)

# Scale back for output
U = scaled_U * z_scale/t_scale
z = scaled_z * z_scale
t = scaled_t * t_scale
fit = [f*z_scale for f in scaled_fit]
misfit = [m*z_scale*z_scale for m in scaled_misfit]

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
