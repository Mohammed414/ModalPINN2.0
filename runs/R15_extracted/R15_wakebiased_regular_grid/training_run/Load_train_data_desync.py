# -*- coding: utf-8 -*-
"""
This file contains functionsspecific to
-Load_train_data_desync.py:
    Python file containing functions that extract and prepare data for training
    and validation.
@author: Gaétan Raynaud
"""

# =============================================================================
# Library import
# =============================================================================

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # required: this codebase uses tf.compat.v1 placeholders
import matplotlib.pyplot as plt
from text_flow import read_flow
from reactions_process import extract_reactions

# =============================================================================
# matplotlib parameters
# =============================================================================

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes',titlesize=20)
plt.rc('legend',fontsize=18)
plt.rc('figure',titlesize=24)


def gen_int_random_points(Npoints,geom,method = 'uniform', disp_plot = False, Delta_r_c = 0.5):
    """
    Generate spatial Navier--Stokes collocation points.

    R12/R14 random wake-biased sampler:
      30% whole-domain random
      35% formation wake random: 0 <= x < 3, |y| <= 2
      25% far wake random:       3 <= x <= 8, |y| <= 2
      10% near-cylinder random annulus: 0.5 < r <= 1.2

    R15 structured wake-biased grid:
      EXACTLY the same 30/35/25/10 source counts and regions, but each
      source component is deterministic and approximately equidistant:
        - Cartesian cell-centred grids in the three rectangular regions;
        - an area-regular staggered polar grid in the annulus.

    The multi-zone construction intentionally retains overlap between source
    components, just as the random wake-biased mixture does. There is no
    importance correction: wake/formation physics is deliberately upweighted.
    """
    print('Generating %d points for equations penalization'%(Npoints))
    print('Method = '+method)

    Lxmin,Lxmax,Lymin,Lymax,x_c,y_c,r_c = geom

    def uniform_box(n, xmin, xmax, ymin, ymax):
        xs = np.empty(n, dtype=np.float64)
        ys = np.empty(n, dtype=np.float64)
        filled = 0
        while filled < n:
            need = n-filled
            m = max(need*2, 64)
            xt = xmin + (xmax-xmin)*np.random.rand(m)
            yt = ymin + (ymax-ymin)*np.random.rand(m)
            ok = ((xt-x_c)**2 + (yt-y_c)**2 > r_c**2)
            xt = xt[ok]
            yt = yt[ok]
            take = min(need, len(xt))
            if take:
                xs[filled:filled+take] = xt[:take]
                ys[filled:filled+take] = yt[:take]
                filled += take
        return xs,ys

    def regular_box_exact(n, xmin, xmax, ymin, ymax):
        # Build a cell-centred Cartesian grid with dx ~ dy, reject the
        # cylinder, and (only if necessary) thin the slight oversupply with
        # evenly distributed deterministic indices. The output count is exact.
        width = float(xmax-xmin)
        height = float(ymax-ymin)
        if n <= 0 or width <= 0.0 or height <= 0.0:
            raise ValueError('Invalid regular_box_exact request.')

        # Start slightly above the ideal area count because the cylinder may
        # remove points. Increase resolution until at least n fluid points exist.
        target = int(np.ceil(1.02*n))
        nx = max(2, int(np.ceil(np.sqrt(target*width/height))))
        ny = max(2, int(np.ceil(target/float(nx))))

        while True:
            xv = xmin + (np.arange(nx)+0.5)*(width/nx)
            yv = ymin + (np.arange(ny)+0.5)*(height/ny)
            X,Y = np.meshgrid(xv,yv,indexing='xy')
            xx = X.ravel()
            yy = Y.ravel()
            ok = ((xx-x_c)**2 + (yy-y_c)**2 > r_c**2)
            xx = xx[ok]
            yy = yy[ok]
            if len(xx) >= n:
                break

            # Refine the direction with the larger current physical spacing.
            dx = width/nx
            dy = height/ny
            if dx >= dy:
                nx += 1
            else:
                ny += 1

        if len(xx) > n:
            # Even thinning across the flattened grid. Unlike random removal,
            # this preserves deterministic broad coverage and does not create
            # a large local hole.
            sel = np.floor(
                (np.arange(n)+0.5)*len(xx)/float(n)
            ).astype(np.int64)
            sel = np.minimum(sel, len(xx)-1)
            xx = xx[sel]
            yy = yy[sel]

        if len(xx) != n:
            raise RuntimeError('regular_box_exact failed to return exact count.')

        return xx.astype(np.float64),yy.astype(np.float64)

    def regular_annulus_exact(n, rin, rout):
        # Find a factorization nr*ntheta=n whose radial and circumferential
        # spacings are as similar as practical.
        best = None
        rmean = 0.5*(rin+rout)
        for nr in range(1,int(np.sqrt(n))+1):
            if n % nr != 0:
                continue
            nt = n//nr
            dr = (rout-rin)/float(nr)
            ds = 2*np.pi*rmean/float(nt)
            score = abs(np.log((dr+1e-30)/(ds+1e-30)))
            if best is None or score < best[0]:
                best = (score,nr,nt)

        _,nr,ntheta = best

        # Equal-area radial locations: r^2 is equally spaced.
        frac = (np.arange(nr)+0.5)/float(nr)
        rr = np.sqrt(rin**2 + frac*(rout**2-rin**2))

        xs=[]
        ys=[]
        dtheta=2*np.pi/float(ntheta)
        for i,rval in enumerate(rr):
            # Stagger neighboring rings by half an angular cell to avoid
            # artificial radial spokes.
            offset = 0.5*dtheta if (i % 2) else 0.0
            th = offset + np.arange(ntheta)*dtheta
            xs.append(x_c + rval*np.cos(th))
            ys.append(y_c + rval*np.sin(th))

        xx=np.concatenate(xs)
        yy=np.concatenate(ys)
        if len(xx) != n:
            raise RuntimeError('regular_annulus_exact count failure.')
        return xx.astype(np.float64),yy.astype(np.float64)

    if method in ('wake_biased','wake_biased_grid'):
        n_whole = int(round(0.30*Npoints))
        n_form  = int(round(0.35*Npoints))
        n_far   = int(round(0.25*Npoints))
        n_ann   = Npoints - n_whole - n_form - n_far

        if method == 'wake_biased':
            x0,y0 = uniform_box(n_whole, Lxmin,Lxmax,Lymin,Lymax)
            x1,y1 = uniform_box(n_form, max(0.0,Lxmin), min(3.0,Lxmax),
                                max(-2.0,Lymin), min(2.0,Lymax))
            x2,y2 = uniform_box(n_far, max(3.0,Lxmin), Lxmax,
                                max(-2.0,Lymin), min(2.0,Lymax))

            r_outer = 1.2
            theta = 2*np.pi*np.random.rand(n_ann)
            rr = np.sqrt(r_c**2 + (r_outer**2-r_c**2)*np.random.rand(n_ann))
            x3 = x_c + rr*np.cos(theta)
            y3 = y_c + rr*np.sin(theta)

        else:
            x0,y0 = regular_box_exact(
                n_whole,Lxmin,Lxmax,Lymin,Lymax)
            x1,y1 = regular_box_exact(
                n_form,max(0.0,Lxmin),min(3.0,Lxmax),
                max(-2.0,Lymin),min(2.0,Lymax))
            x2,y2 = regular_box_exact(
                n_far,max(3.0,Lxmin),Lxmax,
                max(-2.0,Lymin),min(2.0,Lymax))
            x3,y3 = regular_annulus_exact(n_ann,r_c,1.2)

        x = np.concatenate([x0,x1,x2,x3])
        y = np.concatenate([y0,y1,y2,y3])

        if len(x) != Npoints:
            raise RuntimeError('Wake-biased sampler returned wrong point count.')

        # Keep deterministic geometry for the grid case. Random mixture is
        # shuffled as before.
        if method == 'wake_biased':
            perm = np.random.permutation(Npoints)
            x = x[perm]
            y = y[perm]

        print('source mixture counts: whole=%d formation=%d far=%d annulus=%d' %
              (n_whole,n_form,n_far,n_ann))
        if method == 'wake_biased_grid':
            print('R15 structured/equidistant multi-zone grid active.')

    else:
        x = np.zeros(Npoints)
        y = np.zeros(Npoints)

        for j in range(Npoints):
            indomain = False
            while indomain == False:
                x_test = (Lxmax-Lxmin)*np.random.rand(1)[0] + Lxmin

                if method == 'y_normal':
                    y_test = np.random.normal(
                        y_c,0.5*(Lymax-Lymin),1)[0]

                elif method == '2zones' and np.random.uniform()>0.8:
                    r_random = r_c + np.random.uniform()*Delta_r_c
                    theta_random = np.random.uniform()*2*np.pi
                    x_test = x_c + r_random*np.cos(theta_random)
                    y_test = y_c + r_random*np.sin(theta_random)

                else:
                    y_test = (Lymax-Lymin)*np.random.rand(1)[0] + Lymin

                if (((x_test-x_c)**2 + (y_test-y_c)**2 > r_c**2) and
                        y_test < Lymax and y_test > Lymin):
                    x[j] = x_test
                    y[j] = y_test
                    indomain = True

    if disp_plot:
        plt.figure()
        plt.scatter(x,y,c='black',marker='.',s=1.)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Training points: '+method)
        plt.tight_layout()

    return x,y



def read_cut_simulation_data(filename_data,geom):
    '''
    Read simulation data results from filename_data file
    Crop data points in the fluid domain defined by Lxmin,Lxmax,Lymin,Lymax
    ----
    Return Re, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps
    '''
    Lxmin,Lxmax,Lymin,Lymax,x_c,y_c,r_c = geom
    # Step 1 : data loading
    print('Simulation data reading...')
    Re, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps = read_flow(filename_data)

    # Step 2 : Selection of points

    condition_cut_parts = np.array([np.where(nodes_X[0,:] < Lxmax,True,False), \
                  np.where(nodes_X[0,:] > Lxmin,True,False), \
                  np.where(nodes_Y[0,:] > Lymin,True,False), \
                  np.where(nodes_Y[0,:] < Lymax,True,False)])
    condition_cut = np.all(condition_cut_parts,axis=0)

    index_cut = np.argwhere(condition_cut)[:,0]

    # Step 3 : cropping

    nodes_X = nodes_X[:,index_cut]
    nodes_Y = nodes_Y[:,index_cut]
    Us = Us[:,index_cut]
    Vs = Vs[:,index_cut]
    Ps = Ps[:,index_cut]

    print('Reading and cropping simulation data ... ok')

    return Re, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps


def cut_data_index(index_space,nodes_X, nodes_Y, Us, Vs, Ps):
    '''
    return data nodes_X, nodes_Y, Us, Vs, Ps [:,index_space]
    '''
    nodes_X = nodes_X[:,index_space]
    nodes_Y = nodes_Y[:,index_space]
    Us = Us[:,index_space]
    Vs = Vs[:,index_space]
    Ps = Ps[:,index_space]

    return nodes_X, nodes_Y, Us, Vs, Ps


def cut_simu_cylinder_only(geom,nodes_X, nodes_Y, Us, Vs, Ps, n_taps=30):
    '''
    nodes_X, nodes_Y, Us, Vs, Ps : [Ntime,Nelt] array of scalars
    feom : array containing geom info [Lxmin,Lxmax,Lymin,Lymax,x_c,y_c,r_c]
    n_taps : number of pressure taps to select uniformly around the cylinder border
    return data from the points located on the cylinder border
    '''
    Lxmin,Lxmax,Lymin,Lymax,x_c,y_c,r_c = geom

    eps = 1e-5
    r = np.sqrt(np.square(nodes_X[0,:]-x_c) + np.square(nodes_Y[0,:]-y_c))
    delta_r_rc = np.square(r-r_c)

    condition_cut_cyl = np.where(delta_r_rc < eps, True, False)
    index_cylinder = np.argwhere(condition_cut_cyl)[:,0]

    nodes_X, nodes_Y, Us, Vs, Ps = cut_data_index(index_cylinder,nodes_X, nodes_Y, Us, Vs, Ps)

    # Select 30 randomly points
    # np.random.shuffle(index_cylinder)
    # index_cylinder = index_cylinder[:30]

    # Select n_taps points that are the nearest from a uniform disposition over the cylinder
    # endpoint=False so that requesting n_taps genuinely yields n_taps distinct physical
    # locations (endpoint=True would place s=0 and s=1 on the same point on the circle)
    s_lin = np.linspace(0.,1.,n_taps,endpoint=False)
    x_points = x_c + r_c*np.cos(2*np.pi*s_lin)
    y_points = y_c + r_c*np.sin(2*np.pi*s_lin)

    index_reduce = 0*x_points
    index_reduce = np.asarray([int(i) for i in index_reduce])
    for k in range(len(x_points)):
        index_reduce[k] = np.argmin(np.square(nodes_X[0,:] - x_points[k])+np.square(nodes_Y[0,:] - y_points[k]))

    print('Cylinder taps requested: %d, distinct mesh nodes found: %d' % (n_taps, len(np.unique(index_reduce))))

    nodes_X, nodes_Y, Us, Vs, Ps = cut_data_index(index_reduce,nodes_X, nodes_Y, Us, Vs, Ps)

    return nodes_X, nodes_Y, Us, Vs, Ps

def cut_simu_pitot_only(geom,nodes_X, nodes_Y, Us, Vs, Ps):
    '''
    nodes_X, nodes_Y, Us, Vs, Ps : [Ntime,Nelt] array of scalars
    feom : array containing geom info [Lxmin,Lxmax,Lymin,Lymax,x_c,y_c,r_c]
    Return data from the points on pitot points locations
    '''
    Lxmin,Lxmax,Lymin,Lymax,x_c,y_c,r_c = geom
    d = 2*r_c
    # Step 2 : defining the position of wanted points
    N_per_section = 10
    x_points = np.zeros(4*N_per_section)
    y_points = np.zeros(4*N_per_section)

    # line 1
    x_points[:N_per_section] = -3.*d*np.ones(N_per_section)
    y_points[:N_per_section] = np.linspace(Lymin,Lymax,N_per_section)

    # line 2 post cylinder
    x_points[N_per_section:2*N_per_section] = d*np.ones(N_per_section)
    y_points[N_per_section:2*N_per_section] = np.linspace(Lymin,Lymax,N_per_section)

    # line 3
    x_points[2*N_per_section:3*N_per_section] = 2*d*np.ones(N_per_section)
    y_points[2*N_per_section:3*N_per_section] = np.linspace(Lymin,Lymax,N_per_section)

    # line 4
    x_points[3*N_per_section:4*N_per_section] = 3*d*np.ones(N_per_section)
    y_points[3*N_per_section:4*N_per_section] = np.linspace(Lymin,Lymax,N_per_section)

    # Step 3 : finding closest point in data
    index_pitot = 0*x_points
    index_pitot = np.asarray([int(i) for i in index_pitot])
    for k in range(len(x_points)):
        index_pitot[k] = np.argmin(np.square(nodes_X[0,:] - x_points[k])+np.square(nodes_Y[0,:] - y_points[k]))

    nodes_X, nodes_Y, Us, Vs, Ps = cut_data_index(index_pitot,nodes_X, nodes_Y, Us, Vs, Ps)

    return nodes_X, nodes_Y, Us, Vs, Ps


def read_cut_simulation_data_exp_point_and_cylinder(filename_data,geom,n_taps=30):
    '''
    Read simulation data results from filename_data_result
    Pick out data that are the nearest from simulated experimental measurement points
    n_taps : number of pressure taps to select uniformly around the cylinder border
    ----
    Return Re, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps
    '''
    Lxmin,Lxmax,Lymin,Lymax,x_c,y_c,r_c = geom
    # Step 1 : data loading and cut into studied domain [Lxmin,Lxmax]x[Lymin,Lymax]
    Re, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps = read_cut_simulation_data(filename_data,geom)

    data_pitot = cut_simu_pitot_only(geom,nodes_X, nodes_Y, Us, Vs, Ps)
    #data_pitot = [x_pitot, y_pitot, u_pitot_, v_pitot, p_pitot]

    data_cyl = cut_simu_cylinder_only(geom,nodes_X, nodes_Y, Us, Vs, Ps, n_taps=n_taps)
    #data_cyl = [x_cyl, y_cyl, u_cyl, v_cyl, p_cy]

    return times,data_cyl,data_pitot

def read_cut_simulation_data_inlet_points(filename_data,geom):
    '''
    Read simulation data results from filename_data
    Select points from a uniform sampling of location at x = Lxmin and between y = Lymin to y= Lymax on 10 points
    Return times, x_inlet, y_inlet, u_inlet, v_inlet, p_inlet
    where times [Nt,] list of instants
    and x,y,u,v,p are of size [Nt,10]
    '''

    Lxmin,Lxmax,Lymin,Lymax,x_c,y_c,r_c = geom
    Re, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps = read_cut_simulation_data(filename_data,geom)

    # Select 10 points that are the nearest from a uniform disposition over the inlet
    s_lin = np.linspace(0.,1.,10)
    x_points = Lxmin + 0.*s_lin
    y_points = Lymin + s_lin*(Lymax-Lymin)

    index_reduce = 0*x_points
    index_reduce = np.asarray([int(i) for i in index_reduce])
    for k in range(len(x_points)):
        index_reduce[k] = np.argmin(np.square(nodes_X[0,:] - x_points[k])+np.square(nodes_Y[0,:] - y_points[k]))

    x_inlet, y_inlet, u_inlet, v_inlet, p_inlet = cut_data_index(index_reduce,nodes_X, nodes_Y, Us, Vs, Ps)


    return times, x_inlet, y_inlet, u_inlet, v_inlet, p_inlet


def data_flatten_cut(x,y,t,u,v,p,Nmes=0):
    '''
    x,y,t,u,v,p : [Nt,Nelts] array
    Nmes (int) : random truncature of data to Nmes. If Nmes= 0, no truncature is performed
    ----
    return x_ft,y_ft,t_ft,u_ft,v_ft,pft flattened and truncated : 1D array of size [Nmes,] or [Nt*Nelts] if Nmes = 0
    '''

    index = np.array(range(len(x[0,:])*len(x[:,0])))

    if Nmes != 0:
        np.random.shuffle(index)
        index = index[:Nmes]

    x = np.ndarray.flatten(x)[index]
    y = np.ndarray.flatten(y)[index]
    t = np.ndarray.flatten(t)[index]
    u = np.ndarray.flatten(u)[index]
    v = np.ndarray.flatten(v)[index]
    p = np.ndarray.flatten(p)[index]

    return x,y,t,u,v,p

def get_reactions(filename,timemin=-1.,timemax=1e10):
    '''
    Extract forces on cylinder from a .reactions file using extract_reactions()
    Cut time axis between timemin and timemax
    Return times, Fx, Fy
    '''
    times, Fx, Fy, Mz, flag = extract_reactions(filename)

    condition_cut_parts = np.array([np.where(times > timemin, True, False), np.where(times < timemax, True, False)])
    condition_cut = np.all(condition_cut_parts,axis=0)

    index_cut = np.argwhere(condition_cut)[:,0]

    times = times[index_cut]
    Fx = Fx[index_cut]
    Fy = Fy[index_cut]

    return times, Fx, Fy


def addNoise(x,stdNoise):
    """
    x : 1D np array
    stdNoise float > 0. : standard deviation of Gaussian Noise
    Return x + epsilon, epsilon ~ N(0,std)
    """

    return x + np.random.normal(loc=0.0,scale=stdNoise,size=len(x))


def training_dict(Nmes,Nint,Nbc,filename_data,geom,Tintmin=0.,Tintmax=1e2,cut=True,data_selection='all',desync=False,multigrid=False,Ngrid=10,stdNoise=0.,method_int = '2zones',n_taps=30):
    '''
    cut = True : if True, cut data set to only Nmes points
    if False, keep all the values
    ---
    data_selection (str) :
        if 'all' : returns data randomly sampled in fluid domain in quantity Nmes
        if 'inlet' : returns data at 10 points uniformly separated points at inlet
        if 'cylinder_only' : returns data only from points on the border of the cylinder
        if 'cylinder_pitot' : returns data from cylinder border and on pitot points
        if 'pitot_only'  returns data only at pitot points
    desync = False : (bool) if True, add a uniformly distributed phase shift for measuremnts in pitot and cylinder at each position
    multigrid = False (bool) if True, return a list of size Ngrid, each element containing a sampling of space-time coordinates of size Nint
    Ngrid (int) length of the list of sampled space-time coordinates returned when multigrid = True
    '''
    # Part 1 : border normalised coordinate
    s_train = np.random.rand(Nbc)
    # Part 2 : int points

    if multigrid:
        x_int = []
        y_int = []
        t_int = []
        for k in range(Ngrid):
            x_int_temp,y_int_temp = gen_int_random_points(Nint,geom,method = method_int,disp_plot=False)
            x_int.append(x_int_temp)
            y_int.append(y_int_temp)
            t_int.append(Tintmin + (Tintmax-Tintmin)*np.random.rand(Nint))
    else:
        x_int,y_int = gen_int_random_points(Nint,geom,method = method_int,disp_plot=False)
        t_int = Tintmin + (Tintmax-Tintmin)*np.random.rand(Nint)

    # Part 3 : simulation data points


    if data_selection == 'all':

        Re, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps = read_cut_simulation_data(filename_data,geom)
        times_dedouble = np.asarray([t*np.ones(len(nodes_X[0,:])) for t in times])
        if cut:
            xmes,ymes,tmes,umes,vmes,pmes = data_flatten_cut(nodes_X,nodes_Y,times_dedouble,Us,Vs,Ps,Nmes)
        else:
            xmes,ymes,tmes,umes,vmes,pmes = data_flatten_cut(nodes_X,nodes_Y,times_dedouble,Us,Vs,Ps)

        return x_int,y_int,t_int,s_train,xmes,ymes,tmes,umes,vmes,pmes

    elif data_selection == 'inlet':

        times2, x_inlet, y_inlet, u_inlet, v_inlet, p_inlet = read_cut_simulation_data_inlet_points(filename_data,geom)
        t_inlet = np.asarray([t*np.ones(len(x_inlet[0,:])) for t in times2])
        x_inlet, y_inlet, t_inlet, u_inlet, v_inlet, p_inlet = data_flatten_cut(x_inlet, y_inlet, t_inlet, u_inlet, v_inlet, p_inlet)


        return x_int, y_int, t_int, s_train, x_inlet, y_inlet, t_inlet, u_inlet, v_inlet, p_inlet

    else:
        cut = False
        times,data_cyl,data_pitot = read_cut_simulation_data_exp_point_and_cylinder(filename_data,geom,n_taps=n_taps)

        # Cylinder data
        xmes_cyl, ymes_cyl, umes_cyl, vmes_cyl, pmes_cyl = data_cyl
        tmes_cyl = np.asarray([t*np.ones(len(xmes_cyl[0,:])) for t in times])
        xmes_cyl, ymes_cyl, tmes_cyl, umes_cyl, vmes_cyl, pmes_cyl = data_flatten_cut(xmes_cyl, ymes_cyl, tmes_cyl, umes_cyl, vmes_cyl, pmes_cyl)
        pmes_cyl = addNoise(pmes_cyl,stdNoise)

        # Pitot data
        xmes_pitot, ymes_pitot, umes_pitot, vmes_pitot, pmes_pitot = data_pitot

        Nxpitot = 40
        TDesyncMax = 6.06
        if desync:
            #Delta_phi_np_pitot = np.random.uniform(low=0.0,high=TDesyncMax, size=Nxpitot)
            Delta_t_np_pitot = np.random.uniform(low=0.0,high=TDesyncMax, size=Nxpitot)
            tmes_pitot = np.asarray([[times[t] + Delta_t_np_pitot[k] for k in range(len(xmes_pitot[t,:]))] for t in range(len(times))])
        else:
            Delta_t_np_pitot = 0.*np.random.uniform(low=0.0,high=TDesyncMax, size=Nxpitot)
            tmes_pitot = np.asarray([t*np.ones(len(xmes_pitot[0,:])) for t in times])

        xmes_pitot, ymes_pitot, tmes_pitot, umes_pitot, vmes_pitot, pmes_pitot = data_flatten_cut(xmes_pitot, ymes_pitot, tmes_pitot, umes_pitot, vmes_pitot, pmes_pitot)
        umes_pitot = addNoise(umes_pitot,stdNoise)
        vmes_pitot = addNoise(vmes_pitot,stdNoise)
        pmes_pitot = addNoise(pmes_pitot,stdNoise)

        if data_selection == 'cylinder_only':
            return x_int,y_int,t_int,s_train,xmes_cyl,ymes_cyl,tmes_cyl,umes_cyl,vmes_cyl,pmes_cyl

        elif data_selection == 'pitot_only':

            return x_int,y_int,t_int,s_train,xmes_pitot,ymes_pitot,tmes_pitot,umes_pitot,vmes_pitot,pmes_pitot,Delta_t_np_pitot

        elif data_selection == 'cylinder_pitot' :

            return x_int,y_int,t_int,s_train,xmes_pitot,ymes_pitot,tmes_pitot,umes_pitot,vmes_pitot,pmes_pitot,xmes_cyl,ymes_cyl,tmes_cyl,umes_cyl,vmes_cyl,pmes_cyl,Delta_t_np_pitot

        else :

            return x_int,y_int,t_int,s_train

def find_pression_static_amont():
    Re, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps = read_cut_simulation_data(filename_data,geom)
    x_target = -4.
    y_target = -4.
    index = np.argmin(np.square(nodes_X[0,:] - x_target)+np.square(nodes_Y[0,:] - y_target))
    plt.figure()
    plt.plot(times,Ps[:,index])
    plt.xlabel('$t$')
    plt.ylabel('Pressure $p$')
