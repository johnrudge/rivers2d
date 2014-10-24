# Matrix assembly and related routines for 2D river inversion

from dolfin import Mesh, Vertex, Point, Cell, cells, faces, vertices  # leverage some mesh manipulation from dolfin
from numpy import array, zeros, zeros_like, ones_like, where, diff, sqrt, outer, ones, tril_indices, concatenate, arange, meshgrid, argsort, searchsorted, insert, append, tile, nonzero, ndarray, bincount, squeeze, asarray  # basic matrix algebra
from scipy.sparse import coo_matrix, hstack, vstack # sparse matrix
from scipy.optimize import nnls
import glob # file matching
import lbfgs_nnls # non-negative least squares fitting routine
from sklearn.utils.extmath import safe_sparse_dot

def bary_coords(x, y, mesh):
    """Calculate barycentric coordinates of a set of points, returning arrays with vertex nodes and weights"""
    # Work out which cells the points are in
    # ... it would be nice to have a properly vectorized version of this

    try:
        # latest version of dolfin supports this more optimised? method
        bbox = mesh.bounding_box_tree()
        cells = array([bbox.compute_first_entity_collision(Point(xi,yi)) for xi,yi in zip(x,y)]) 
    except:
        cells = array([mesh.intersected_cell(Point(xi,yi)) for xi,yi in zip(x,y)]) 
    
    vindices = mesh.cells()[cells]     # indices of all the vertices
    coo = mesh.coordinates()[vindices] # coordinates of all the vertices
    coo_x = coo[:,:,0]                 # x-coordinates of all the vertices
    coo_y = coo[:,:,1]                 # y-coordinates of all the vertices
    
    # Vector from point to one of the opposing points
    d_x = coo_x[:,(1,2,0)] - x[:,None]
    d_y = coo_y[:,(1,2,0)] - y[:,None]
    
    # Vector between two opposing points
    e_x = coo_x[:,(1,2,0)] - coo_x[:,(2,0,1)]
    e_y = coo_y[:,(1,2,0)] - coo_y[:,(2,0,1)]
    
    # Vector from point to one of opposing points
    f_x = e_x[:,(1,2,0)]
    f_y = e_y[:,(1,2,0)]
    
    weights = abs((e_x*d_y - d_x *e_y) / (e_x*f_y - e_y*f_x))  # ratio of cross-products
    
    return vindices, weights


def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


def trapezoidal_rule_weights(s, axis = -1):
    """Given a sequence of times, calculate the appropriate weights of the trapezoidal rule"""
    d = diff(s, axis=axis)
    nd = len(s.shape)
    
    slice1 = tupleset((slice(None),)*nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),)*nd, axis, slice(None, -1))
    
    res = zeros_like(s)
    res[slice1] += 0.5*d
    res[slice2] += 0.5*d
    
    return res


def assemble_space_smooth(mesh, t):
    """Build matrix and rhs for smoothing in space"""
    print 'Begin spatial smoothing assemble...'
    nt = len(t)
    nv = mesh.num_vertices()
    nf = mesh.num_faces()

    sm_con = zeros((nf,3)) # list of spatial smoothing connections
    sm_x = zeros((nf,3))   # weights for gradient in x
    sm_y = zeros((nf,3))   # weights for gradient in y
    for idx, face in enumerate(faces(mesh)):
        area = face.area()
        sm_con[idx,:] = [v.index() for v in vertices(face)]    # vertex indices 
        x = [v.point().x() for v in vertices(face)]  # vertex x-coordinates
        y = [v.point().y() for v in vertices(face)]  # vertex y-coordinates
        
        # Calculate gradient of U weighted by sqrt(area)
        sm_x[idx,:] = array([y[1]-y[2], y[2]-y[0], y[0]-y[1]]) / (2.0*sqrt(area))
        sm_y[idx,:] = array([x[2]-x[1], x[0]-x[2], x[1]-x[0]]) / (2.0*sqrt(area))

    n_spacesmooth = nf*3*nt*2
    print 'Spatial smoothing entries = ', n_spacesmooth

    # weight times according to trapezoidal rule
    tweight = sqrt(trapezoidal_rule_weights(t))
    
    i, j, k = meshgrid(arange(nf), arange(nt), arange(3))
    row  = i + j*nf
    col  = sm_con[i,k] + j*nv
    datx = sm_x[i,k] * tweight[j]
    daty = sm_y[i,k] * tweight[j]

    Sx = coo_matrix((datx.flat,(row.flat,col.flat)),shape =(n_spacesmooth/2, nv*nt))
    Sy = coo_matrix((daty.flat,(row.flat,col.flat)),shape =(n_spacesmooth/2, nv*nt))
    
    S = vstack([Sx,Sy],'csc')
    b = zeros(n_spacesmooth)
    
    print "Space smooth assembled."
    return S, b


def assemble_time_smooth(mesh, t):
    """Build matrix and rhs for smoothing in time"""
    print 'Begin temporal smoothing assemble...'
    # temporal smoothing
    nt = len(t)
    nv = mesh.num_vertices()
    n_timesmooth = nv*(nt-1)
    print 'Temporal smoothing entries = ', n_timesmooth

    vidx = zeros(nv)
    a_weight = zeros(nv)
    for i, v in enumerate(vertices(mesh)):
        # Calculate total area of cells touched by given vertex
        area_int = 0
        for c in cells(v):
            area_int += c.volume() 
        a_weight[i] = sqrt(area_int/3.0)
        vidx[i] = v.index()

    t_weight = 1.0/ sqrt(diff(t))
 
    i, j = meshgrid(arange(nv), arange(nt-1))

    rows = i + j *nv
    colsm = vidx[i] + j *nv
    colsp = vidx[i] + (j+1) *nv
    datm = a_weight[i] * t_weight[j]
    datp = -a_weight[i] * t_weight[j]
    
    row = concatenate([rows.flat, rows.flat])
    col = concatenate([colsm.flat, colsp.flat])
    dat = concatenate([datm.flat, datp.flat])
    
    S = coo_matrix((dat,(row,col)),shape =(n_timesmooth, nv*nt))
    S = S.tocsc()
    b = zeros(n_timesmooth)
 
    print "Time smooth assembled."
    return S, b


def bound_array(arr, minv, maxv, axis = -1):
    """Take an array, and bound it between some max and min values"""
    bound_arr = arr
    
    if isinstance(minv, ndarray):
        nd = len(arr.shape)    
        tset = tupleset((None,)*nd, axis, slice(None))

        minidx = (arr < minv[tset]).nonzero()
        maxidx = (arr > maxv[tset]).nonzero()

        bound_arr[minidx] = minv[minidx[axis]]
        bound_arr[maxidx] = maxv[maxidx[axis]]
    else:
        bound_arr[arr<minv] = minv
        bound_arr[arr>maxv] = maxv

    return bound_arr


def time_block_splits(a, b, b_sortidx):
    """Work out nearest time indices and time block split"""
    ir = arange(b.shape[0])
    ii = ir[:,None]
    b_sorted = b[ii, b_sortidx]  # sort b

    # Work out where each value of a could be inserted into b_sorted to preserve order
    a_left_bidx = array([searchsorted(b_sorted[i,:], a[i,:], side='right')-1 for i in ir])
    a_right_bidx = array([searchsorted(b_sorted[i,:], a[i,:], side='left') for i in ir])

    # Bound arrays so don't go off ends
    a_left_bidx = bound_array(a_left_bidx, 0, b.shape[1]-1)
    a_right_bidx = bound_array(a_right_bidx, 0, b.shape[1]-1)

    # Work out alpha, how close you are to each neighbouring point
    a_alpha = zeros_like(a)
    bl = b_sorted[ii, a_left_bidx]
    br = b_sorted[ii, a_right_bidx]
    cond = ~(a_left_bidx == a_right_bidx) # conditional to avoid divide by zero when match
    a_alpha[cond] = (a[cond] - bl[cond]) / (br[cond]-bl[cond])
    
    # convert from sorted indices back to original
    a_left_bidx = b_sortidx[ii, a_left_bidx]
    a_right_bidx = b_sortidx[ii, a_right_bidx]
    
    return a_left_bidx, a_right_bidx, a_alpha    


def assemble_model_single_river(mesh, t, river_dict):
    """Assemble the model matrix and rhs for a single river""" 
    rt = river_dict['t']
    rx = river_dict['x']
    ry = river_dict['y']
    rz = river_dict['z']
    nt = len(t)
    nv = mesh.num_vertices()

    b = rz[1:]  # rhs 

    # Calculate times in past on characteristic curve from Gilbert times
    tm = rt[:,None] - rt[None,:]
    
    # Create a complete list of time nodes, both Gilbert and uplift
    tv = tile(t, (len(rz),1))
    s = concatenate([tm , tv], axis=1) 
    
    # Sort s into ascending order, and figure out mapping between s and original times
    sidx = argsort(s, axis=1)  # indices which sort rows of s into increasing order
    i = arange(len(rt))
    ii = i[:,None]
    s = s[ii,sidx] # sort s
    idx = zeros_like(sidx)
    idx[ii,sidx] = arange(sidx.shape[1]) # idx is reverse lookup of sidx   
    idx_tm = idx[:,0:len(rt)]    # Note tm[i,:] = s[i, idx_tm[i,:]]
    idx_t = idx[:,len(rt):]      # Note t = s[i, idx_t[i,:]]
    
    # Calculate weights for the trapezoidal rule
    s_bounded = bound_array(s, array([rt[0]]*len(rt)), rt, axis = 0) # set range of integration
    ds = trapezoidal_rule_weights(s_bounded, axis = 1)
    ds_tm = ds[ii, idx_tm]
    ds_t = ds[ii, idx_t]
    
    # Construct lookups from Gilbert to nearest uplift nodes and vice-versa needed for interpolation
    tv_sortidx = tile(arange(len(t)), (len(rz),1)) # indices which sort tv
    tm_sortidx = tile(arange(len(rt))[::-1], (len(rz),1)) # indices which sort tm   
    t_left_tm_idx, t_right_tm_idx, t_tm_alpha = time_block_splits(tv, tm, tm_sortidx)
    tm_left_t_idx, tm_right_t_idx, tm_t_alpha = time_block_splits(tm, tv, tv_sortidx)

    ### ***Contributions from Gilbert time nodes***
    # Work out indices that have a contribution
    i, j = nonzero(ds_tm)   # i is index of start point, j are indices along integration path    
    dsij = ds_tm[i,j]       # trapezoidal rule weights for these indices

    # Work out barycentric co-ordinates of each point on river
    vs, ws = bary_coords(rx, ry, mesh)
    ii = i[:,None]
    jj = j[:,None]
    ds = dsij[:,None]
    kk = arange(3)[None,:]
    
    # Work out contributions, linearly interpolating between uplift time nodes
    rows_left = rows_right = tile(ii - 1, 3)
    
    cols_left = vs[jj,kk] + tm_left_t_idx[ii,jj]*nv
    dats_left = ds * ws[jj,kk] * (1.0-tm_t_alpha[ii,jj])
    
    cols_right = vs[jj,kk] + tm_right_t_idx[ii,jj]*nv
    dats_right = ds * ws[jj,kk] * tm_t_alpha[ii,jj]
    
    ### ***Contributions from uplift time nodes***
    # Work out indices that have a contribution
    i, j = nonzero(ds_t)   # i is index of start point, j are indices along integration path    
    dsij = ds_t[i,j]       # trapezoidal rule weights for these indices

    # Use linear interpolation between Gilbert time nodes to find positions at uplift time nodes
    x = rx[t_left_tm_idx[i,j]] * (1.0-t_tm_alpha[i,j]) + rx[t_right_tm_idx[i,j]] * t_tm_alpha[i,j]
    y = ry[t_left_tm_idx[i,j]] * (1.0-t_tm_alpha[i,j]) + ry[t_right_tm_idx[i,j]] * t_tm_alpha[i,j]
   
    # Work out barycentric co-ordinates of each interpolation point
    vs, ws = bary_coords(x, y, mesh)
    ii = i[:,None]
    jj = j[:,None]
    ds = dsij[:,None]
    
    rows_uplift = tile(ii - 1, 3)
    cols_uplift = vs + jj * nv
    dats_uplift = ds * ws
    
    # Build final matrix, exploiting fact coo_matrix automatically sums
    row = concatenate([rows_left.flat, rows_right.flat, rows_uplift.flat])
    col = concatenate([cols_left.flat, cols_right.flat, cols_uplift.flat])
    dat = concatenate([dats_left.flat, dats_right.flat, dats_uplift.flat])

    M = coo_matrix((dat,(row,col)),shape =(len(b), nv*nt))
    M = M.tocsc()
    return M, b


def assemble_model(mesh, t, river_data):
    """Assemble model matrices and rhs for a list of rivers"""
    print 'Begin model assemble...'
    
    Ms = []
    bs = []

    for ifile, river_dict in enumerate(river_data):
        print 'Processing %s [%d/%d]'%(river_dict['filename'], ifile + 1, len(river_data))
        M, b = assemble_model_single_river(mesh, t, river_dict)
        Ms.append(M)
        bs.append(b)

    print "Model assembled."
    return Ms, bs


def parse_river_file(filename, v, m):
    """Parse a single river file"""
    print "Reading : ", filename
    f = open(filename)
    riverdata = f.readlines()
    f.close()

    # Parse river data
    rx = []
    ry = []
    rz = []
    A = []
    d = []
    for line in riverdata:
        b = line.split(" ")
        rx.append(float(b[0])/1000.0)
        ry.append(float(b[1])/1000.0)
        rz.append(float(b[2]))
        d.append(float(b[3]))
        A.append(float(b[4]))

    # reorder arrays from river mouth instead of from head
    rx = array(rx)[::-1]
    ry = array(ry)[::-1]
    rz = array(rz)[::-1]
    A = array(A)[::-1]
    d = array(d)[::-1]

    # scan along river, calculating characteristic travel times
    rt = zeros_like(rz)
    
    for i in range(1,len(rz)):
        # Calculate characteristic time to this point
        threshold = 100000.0
        if (A[i] > threshold):
            # Trapezoidal rule
            dx = d[i-1] - d[i]
            rt[i] = rt[i-1] + 0.5*dx*(1.0/(v*A[i-1]**m) + 1.0/(v*A[i]**m))
        else:
            # Upstream area is zero or small - try midpoint rule instead
            print "Warning, switching to midpoint rule for point", i
            dx = d[i-2] - d[i]
            rt[i] = rt[i-2] + dx*(1.0/(v*A[i-1]**m))

    river_dict = {'filename': filename, 'x': rx, 'y': ry, 'z': rz, 't': rt, 'A': A, 'd': d}
    return river_dict


def parse_river_data(input_dir, pattern, v, m):
    """Parse a directory of river files conforming to pattern, returning a list with the data"""
    river_data = [parse_river_file(f, v, m) for f in glob.glob(input_dir + pattern)]
    return river_data


def lsq_inversion(Ms, bs, weights, algorithm, pins):
    """Least squares inversion for uplift"""
    print 'Begin least squares fit'
    
    weighted_Ms = [w*M for w, M in zip(weights, Ms)]
    weighted_bs = [w*b for w, b in zip(weights, bs)]
    
    M = vstack(weighted_Ms,'csc')
    b = hstack(weighted_bs)

    indices = where(~pins)[0]
    M_pin = M[:,indices]  # remove pinned points from matrix

    if algorithm == 'lsqr':
        result_pin = lsqr(M_pin,b)[0]
    
    if algorithm == 'nnls_bfgs':
        nnls_bfgs = lbfgs_nnls.LbfgsNNLS()
        nnlsfit = nnls_bfgs.fit(M_pin, b)
        result_pin = nnlsfit.coef_
        print type(result_pin)
        
    if algorithm == 'nnls_kkt':
        # VERY SLOW way of doing nnls
        # Has to form large dense array
        M_pin_dense = M_pin.todense()
        b_vector= squeeze(asarray(b.todense()))        
        result_pin = nnls(M_pin_dense, b_vector)[0]

    result = zeros(len(pins))
    result[~pins] = result_pin

    fit = [safe_sparse_dot(M,result) for M in Ms]
    residual = [f - b for f, b in zip(fit, bs)]
    misfit = [safe_sparse_dot(r,r) for r in residual] 

    print 'Finished least squares fit'
    return result, fit, residual, misfit


def calculate_nullity(Ms):
    print 'Calculating nullity'
    # Get nullity
    M = vstack(Ms,'csc')
    row, col = M.nonzero()
    nullity = bincount(col, minlength = M.shape[1])
        
    return nullity
