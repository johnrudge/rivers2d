import os, os.path

def ensure_dir(filename):
    """ Ensure directory for filename exists"""
    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)    


def output_fits_ascii(river_data, fit, output_dir):
    
    ensure_dir(output_dir)

    # Each river should just appear in order in 'fit'
    # It would be nice to have the x,y coords here too

    for ix, river_dict in enumerate(river_data):
        rz = river_dict['z']
        filename = river_dict['filename']
        output_file = output_dir + os.path.basename(filename) + "-fit.txt"
        fout = open(output_file,"w")
        for i in range(1,len(rz)):
            fout.write("%f %f\n"%(rz[i], fit[ix][i-1]))
        fout.close()


# Simply output the uplift at all points, at each time
def output_vector_ascii(mesh, t, result, filename):
    from dolfin import Vertex
    ensure_dir(filename)
    fout = open(filename, "w")

    nt, nv = result.shape
    
    for i in range(nt):
        values = result[i,:]
        fout.write("# Output timestep %d, t=%f\n"%(i,t[i]))
        fout.write("# %f\n"%t[i])
        for j in range(nv):
            p = Vertex(mesh, j).point()
            fout.write("%f %f %r\n"%(p.x(), p.y(), values[j]))
        fout.write("\n")
    fout.close()

    return


def gnuplot_write_plot(mesh, values, filename, mode):
    from dolfin import Vertex
    ensure_dir(filename)
    fout = open(filename, "w")
    for i in range(mesh.num_cells()):
        v = mesh.cells()[i]
        # use with splot in gnuplot
        if(mode=='grid'):
            for j in [0,1,2,0]:
                p = Vertex(mesh, v[j]).point()
                fout.write("%f %f %f\n"%(p.x(), p.y(), values[v[j]]))
            # use with set pm3d in gnuplot
        elif(mode=='pm3d'):
            for j in [0,1]:
                p = Vertex(mesh, v[j]).point()
                fout.write("%f %f %f\n"%(p.x(), p.y(), values[v[j]]))
            fout.write("\n")
            for j in [2,2]:
                p = Vertex(mesh, v[j]).point()
                fout.write("%f %f %f\n"%(p.x(), p.y(), values[v[j]]))
        fout.write("\n\n")
    fout.close()


def output_vector_vtk(mesh, t, vector, filename):
    from dolfin import File, FunctionSpace, Function
    ensure_dir(filename)
    f = File(filename)
    Q = FunctionSpace(mesh, "CG", 1)
    F = Function(Q)
    
    try:
        # dolfin 1.3 or higher syntax
        from dolfin import dof_to_vertex_map
        map = dof_to_vertex_map(Q)
    except:
        # dolfin 1.2 syntax
        map = Q.dofmap().vertex_to_dof_map(mesh)
    
    nt, nv = vector.shape

    for i in range(nt):
        v = vector[i,map]
        try:
            F.vector()[:] = v
        except:
            # dolfin 1.5 syntax change?
            from numpy import array
            v = array(v, dtype='float_')
            F.vector().set_local(v)
        f << (F, t[i])


def output_river_data_vtk(river_data, filename):
    try:
        from evtk.hl import pointsToVTK
    except ImportError:
        print("Warning: Python EVTK module not found - not writing VTK river data.")
        return
        
    from numpy import concatenate, zeros_like

    ensure_dir(filename)
    
    # Combine all the river data into single arrays
    xs = []
    ys = []
    zs = []
    ts = []
    for river_dict in river_data:
        xs.append(river_dict['x'])
        ys.append(river_dict['y'])
        zs.append(river_dict['z'])
        ts.append(river_dict['t'])
    x = concatenate(xs)
    y = concatenate(ys)
    z = concatenate(zs)
    t = concatenate(ts)
        
    # Output the data
    pointsToVTK(filename, x, y, zeros_like(z), data = {"Elevation" : z, "Gilbert time" : t})


def output_vector_gnuplot(mesh, t, result, filename, mode='grid'):
    ensure_dir(filename)
    nt, nv = result.shape

    minval = min(result.flat)
    maxval = max(result.flat)

    for i in range(nt):
        values = result[i,:]
        gnuplot_write_plot(mesh, values, "%s_%02d"%(filename,i), mode)

    # Create a run file for gnuplot
    fout = open(filename + "_run","w")
    fout.write('#load this into gnuplot with load "%s_run"\n'%filename)
    fout.write("set zrange [%f:%f]\n"%(minval, maxval))
    if(mode=='pm3d'):
        fout.write("set pm3d\n")
        fout.write("set cbrange [%f:%f]\n"%(minval, maxval))
    fout.write("set style data lines\n")
    for i in range(nt):
        fout.write('splot "%s_%02d"\n'%(filename,i))
        fout.write("pause 1\n")
    fout.write('pause -1 "hit enter to continue"\n')
    fout.close()


def output_mesh_ascii(mesh, filenamebase):
    ensure_dir(filenamebase)

    coordinates_file = filenamebase + "_coords.xyz"
    fout = open(coordinates_file,"w")
    coo = mesh.coordinates()
    for i in range(coo.shape[0]):
        fout.write("%f\t%f\n"%tuple(coo[i,:]))
    fout.close()
    
    connectivity_file = filenamebase + "_connectivity.ijk"
    fout = open(connectivity_file,"w")
    vindices = mesh.cells()
    for i in range(vindices.shape[0]):
        fout.write("%d\t%d\t%d\n"%tuple(vindices[i,:]))
    fout.close()
    
    
    
