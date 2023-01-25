from multiprocessing import Pool

import cartopy.crs as ccrs
from cartopy.feature import LAND, COASTLINE
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
import numpy as np
from scipy.ndimage import zoom, median_filter, gaussian_filter, distance_transform_edt, minimum_filter
from scipy.interpolate import griddata
from scipy.signal import convolve2d
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
from skimage import morphology
from sklearn.feature_extraction.image import img_to_graph
from sklearn.cluster import AgglomerativeClustering

from stereoid.sea_ice import RadarModel

GLOBAL_DATA = None


def read_nextsim_data(filename, proj, lon_0=0):
    """
    Read data from neXtSIM mesh
    
    Inputs:
        filename : str, input NPZ file
        proj : pyproj.proj.Proj, projection for conversion of lon,lat to X,Y
        lon_0 : float, offset to correct 1-360 range of longitues
    Returns:
        x : np.array [n_nodes], projected X coordinate of neXtSIM mesh nodes
        y : np.array [n_nodes], projected Y coordinate of neXtSIM mesh nodes
        v_e: np.array [n_nodes], eastward velocity on neXtSIM mesh nodes
        v_n: np.array [n_nodes], northwardward velocity on neXtSIM mesh nodes
        c : np.array [n_elements], concentration on neXtSIM element
        t : np.array [3, n_elements], 3 indeces of nodes for each element
        
    """
    n = np.load(filename)
    lat = n['lat']
    lon = n['lon']
    u = n['u']
    v = n['v']
    v_e = u * np.cos(np.deg2rad(lon + lon_0)) + v * np.sin(np.deg2rad(lon + lon_0))
    v_n = -u * np.sin(np.deg2rad(lon + lon_0)) + v * np.cos(np.deg2rad(lon + lon_0))
    c = n['Concentration']
    t = n['i']
    x, y = proj(lon, lat)
    return x, y, v_e, v_n, c, t

def create_swath_grids(swth, proj, t_res, y_res, min_lat=68, V=7200):
    """
    Create grids with X, Y and north direction in SWATH projection
    
    Inputs:
        swath : SingleSwathBistatic, swath definition
        proj : pyproj.proj.Proj, projection for conversion of lon,lat to X,Y
        t_res : float, along-track time resolution of generated swath
        y_res : float, Y-axiss resolution
        min_lat : how far soth swath grid can go
        V : float, ???
    Returns:
        x_sz: np.array [cols, rows], X coordinates in proj projection
        y_sz: np.array [cols, rows], Y coordinates in proj projection
        nor_sz: np.array [cols, rows], direction to north

    """
    nor_s = np.array(swth.master_swath.Northing)
    nor_s[nor_s < 0] += 2*np.pi
    lat_s = np.array(swth.master_swath.lat)
    lon_s = np.array(swth.master_swath.lon)

    gpi = lat_s.mean(axis=1) > min_lat
    nor_s = nor_s[gpi]
    lat_s = lat_s[gpi]
    lon_s = lon_s[gpi]
    x_s, y_s = proj(lon_s, lat_s)
    
    y_factor = t_res * V / y_res
    x_sz, y_sz, nor_sz = [zoom(a, (y_factor, 1), order=1) for a in [x_s, y_s, nor_s]]
    return x_sz, y_sz, nor_sz

def interpolate_nextsim_on_swath(x, y, v_e, v_n, c, t, x_sz, y_sz, min_conc=0.5):
    """
    Create grids with nextsim U, V, landmask and ice mask in SWATH projection
    
    Inputs:
        x : np.array [n_nodes], projected X coordinate of neXtSIM mesh nodes
        y : np.array [n_nodes], projected Y coordinate of neXtSIM mesh nodes
        v_e: np.array [n_nodes], eastward velocity on neXtSIM mesh nodes
        v_n: np.array [n_nodes], northwardward velocity on neXtSIM mesh nodes
        c : np.array [n_elements], concentration on neXtSIM element
        t : np.array [3, n_elements], 3 indeces of nodes for each element
        x_sz: np.array [cols, rows], X coordinates in proj projection
        y_sz: np.array [cols, rows], Y coordinates in proj projection
        nor_sz: np.array [cols, rows], direction to north
    Returns:
        v_e_sz: np.array [cols, rows], eastawrd velocity
        v_n_sz: np.array [cols, rows], northward velocity
        landmask: np.array [cols, rows], landmask coordinates
        icemask: np.array [cols, rows], icemask coordinates
        
    """
    v_e_sz = griddata((x, y), v_e, (x_sz.flatten(), y_sz.flatten()), method='nearest').reshape(x_sz.shape)
    v_n_sz = griddata((x, y), v_n, (x_sz.flatten(), y_sz.flatten()), method='nearest').reshape(x_sz.shape)
    # create linear interpolator to get mask of land and low conentration
    tria = Triangulation(x, y, t)
    f = LinearTriInterpolator(tria, x)
    landmask = np.isnan(f(x_sz, y_sz))
    tria = Triangulation(x, y, t, mask=(c < min_conc))
    f = LinearTriInterpolator(tria, x)
    icemask = np.isnan(f(x_sz, y_sz))
    return v_e_sz, v_n_sz, landmask, icemask

def compute_nextsim_uv(v_e_sz, v_n_sz, landmask, nor_sz, med_filt_size=7):
    """
    Compute range and azimuth components of U and V
    
    Inputs:
        v_e_sz: np.array [cols, rows], eastawrd velocity
        v_n_sz: np.array [cols, rows], northward velocity
        landmask: np.array [cols, rows], landmask coordinates
        nor_sz: np.array [cols, rows], direction to north
        med_filt_size: int, size of median filter to smooth nextsim U,V
    Return:
        v_x_sz: np.array [cols, rows], range component of velocity
        v_y_sz: np.array [cols, rows], azimuth component of velocity

    """
    cosn_sz = np.cos(nor_sz)
    sinn_sz = np.sin(nor_sz)
    v_x_sz = v_e_sz * cosn_sz - v_n_sz * sinn_sz
    v_y_sz = v_n_sz * cosn_sz + v_e_sz * sinn_sz
    v_x_sz = median_filter(v_x_sz, med_filt_size)
    v_y_sz = median_filter(v_y_sz, med_filt_size)
    v_x_sz[landmask] = np.nan
    v_y_sz[landmask] = np.nan
    return v_x_sz, v_y_sz


def get_doppler(sgm, obs_geo, u_int, v_int, x_res, pol, inc_m, fstr_s1, fstr_ati, prod_res, b_ati):
    """
    Get radar observed Doppler. [From "E2E nExtSIM to Harmony observed drift velocities.ipynb"]
    """
    x_range_vec = np.arange(u_int.shape[1]) * x_res
    nrcs, dopp = sgm.run_forward_model(obs_geo, pol, u_int, v_int, x_range_vec, inc_m) # NRCS and signal Doppler
    obs_geo.concordia.set_swath(inc_m, x_range_vec)
    obs_geo.discordia.set_swath(inc_m, x_range_vec)
    radarm = RadarModel(obs_geo.concordia, fstr_s1, fstr_ati, fstr_ati, prod_res=prod_res, b_ati=b_ati)
    s_dopp = radarm.sigma_dop(10 ** (nrcs/10)) # Doppler noise standard deviation
    s_dopp_rnd = s_dopp*np.random.randn(dopp.shape[0],dopp.shape[1],3) # NESZ (profile + speckle)
    r_dopp_r0 = dopp + s_dopp_rnd # radar observed Doppler
    
    return dopp, s_dopp, r_dopp_r0

def fill_nan_gaps(array, mask, distance=5):
    dist, indi = distance_transform_edt(
        mask,
        return_distances=True,
        return_indices=True)
    gpi = dist <= distance
    r,c = indi[:,gpi]
    array2 = np.array(array)
    array2[gpi] = array[r,c]
    return array2

def remove_texture_noise(r_dopp, s_dopp, noise_factor=0.7, gf_size=10):
    """ Remove texture noise [Park et al., 2019] """
    noise = s_dopp - s_dopp.min(axis=(0,1))
    noise /= noise.max(axis=(0,1))
    noise *= 0.7
    r_dopp_d = np.array(r_dopp)
    for i in range(3):
        r_dopp_i = r_dopp[:,:,i]
        r_dopp_f = fill_nan_gaps(r_dopp_i, np.isnan(r_dopp_i), 200)
        r_dopp_g = gaussian_filter(r_dopp_f, gf_size, truncate=2)
        r_dopp_g[np.isnan(r_dopp_i)] = np.nan
        r_dopp_d[:,:,i] = r_dopp_g*noise[:,:,i] + r_dopp_i*(1 - noise[:,:,i])

    return r_dopp_d

def get_power_spectrum_mk(v, m=10):
    """ Get power spectrum [Kleinherenbrink et al., 2021] """
     # split so we can do some kind of Bartlett in the along-track direction
    shp=v.shape
    l=int(np.floor(shp[0]/m))
    Pv = np.zeros(shp)
    #Pun=np.zeros(shp); Pvn=np.zeros(shp)
    for i in range(0, m):
        # make sure the splitted data has the same size (zero-padding)
        v_temp=np.zeros(shp)
        # split data
        v_temp[i*l:(i+1)*l,:] = v[i*l:(i+1)*l,:]
        # periodograms
        PSD_v = np.absolute(np.fft.fft2(v_temp))**2/l/shp[1]
        # mean periodogram
        Pv += PSD_v
    for i in range(0,m-1):
        # make sure the splitted data has the same size (zero-padding)
        v_temp=np.zeros(shp)
        # split data
        v_temp[int(i+0.5)*l:(int(i+0.5)+1)*l,:]=v[int(i+0.5)*l:(int(i+0.5)+1)*l,:]
        # periodograms
        PSD_v=np.absolute(np.fft.fft2(v_temp))**2/l/shp[1]
        # mean periodogram
        Pv=Pv+PSD_v
    Pv = Pv/(2*m-1)
    return Pv

def denoise_mk(v1, v2, alpha=0.5):
    """ Remove high frequency noise [Kleinherenbrink et al., 2021] """
    v1c = v1 - np.mean(v1)
    v2c = v2 - np.mean(v2)
    vn = v2c - v1c
    Pv2 = get_power_spectrum_mk(v2c)
    Pve = Pv2 - np.std(vn)**2*alpha
    Pve[Pve < 0] = np.nan
    V2 = np.fft.fft2(v2c)
    Pve[np.isnan(Pve)] = 0
    v2fc = np.real(np.fft.ifft2(Pve / Pv2 * V2))
    v2f = v2fc + np.mean(v2)
    return v2f

def apply_anisotropic_diffusion(img, gamma=0.25, step=(1., 1.), ploton=False, kappa = 50, niter = 10, option = 2):
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal

    Sep 2017 modified by Denis Demchev
    """

    # init args
    
    # Conduction coefficient
    
    
    # Number of iterations
    
    
    # Number of equation (1,2)

    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl

        fig = pl.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(img, interpolation='nearest')
        ih = ax2.imshow(imgout, interpolation='nearest', animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in range(niter):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS / kappa) ** 2.) / step[0]
            gE = np.exp(-(deltaE / kappa) ** 2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[0]
            gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

        if ploton:
            iterstring = "Iteration %i" % (ii + 1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return imgout

def conv2_mk(x, y, mode='same'): # this makes it the same as matlab
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def convolution_filter_mk(vf, size1=15, size2=16):
    shp=vf.shape
    # discontinuities using a n x m filter
    Sv1=np.ones(vf.shape)
    Sv2=np.ones(vf.shape)
    for i in range(size1,size2,2):
        n=int(i)
        m=int(i)

        # filter
        f1=np.vstack((np.ones((n,m)), -1*np.ones((n,m))))/n/m/2
        f2=np.column_stack((-1*np.ones((m,n)), np.ones((m,n))))/n/m/2
        vf1=(conv2_mk(vf,f1,'same'))
        vf2=(conv2_mk(vf,f2,'same'))
        Sv1=Sv1*vf1
        Sv2=Sv2*vf2
    Sv=np.sqrt(Sv1**2+Sv2**2)
    return Sv, Sv1, Sv2

def get_edges_mk(S, thr):
    shp = S.shape
    min_width=3
    max_width=100
    Iy=[]
    Ix=[]
    # go through columns
    for i in range(0,shp[1]):
        Id=np.where(np.log(np.absolute(S[:,i]))>thr)
        if len(Id[0]) > 1:
            dId=Id[0][1:]-Id[0][:-1]
            I1=np.where(dId > 1)
            I1=np.append(0,I1[0]+1)
            I2=np.append(I1[1:],len(Id[0]))
            for k in range(0,len(I1)):
                if Id[0][I2[k]-1]-Id[0][I1[k]] > min_width-1:
                    if Id[0][I2[k]-1]-Id[0][I1[k]] < max_width+1:
                        #I_in=Id[0][I1[k]:I2[k]]
                        #I_out=np.argmax(np.absolute(S[I_in,i]))
                        #Iy=np.append(Iy,I_in[I_out])
                        Iy=np.append(Iy,int(np.mean(Id[0][I1[k]:I2[k]])))
                        Ix=np.append(Ix,i)
    # go through rows
    for i in range(0,shp[0]):
        Id=np.where(np.log(np.absolute(S[i,:]))>thr)
        if len(Id[0]) > 1:
            dId=Id[0][1:]-Id[0][:-1]
            I1=np.where(dId > 1)
            I1=np.append(0,I1[0]+1)
            I2=np.append(I1[1:],len(Id[0]))
            for k in range(0,len(I1)):
                if Id[0][I2[k]-1]-Id[0][I1[k]] > min_width-1:
                    if Id[0][I2[k]-1]-Id[0][I1[k]] < max_width+1:
                        #I_in=Id[0][I1[k]:I2[k]]
                        #I_out=np.argmax(np.absolute(S[i,I_in]))
                        #Ix=np.append(Ix,I_in[I_out])
                        Ix=np.append(Ix,int(np.mean(Id[0][I1[k]:I2[k]])))
                        Iy=np.append(Iy,i)
    return Ix,Iy

def get_edges_x_y_mk(uf, vf, size1=15, size2=16, thr=-4.5):
    Su, Su1, Su2 = convolution_filter_mk(uf, size1, size2)
    Sv, Sv1, Sv2 = convolution_filter_mk(vf, size1, size2)
    X=[]
    Y=[]
    St=np.maximum(np.maximum(np.maximum(np.absolute(Su1),np.absolute(Su2)),np.absolute(Sv1)),np.absolute(Sv2))
    Ix,Iy=get_edges_mk(Su,thr)
    X=np.append(X,Ix)
    Y=np.append(Y,Iy)
    Ix,Iy=get_edges_mk(Sv,thr)
    X=np.append(X,Ix)
    Y=np.append(Y,Iy)
    return X, Y

def morph_operators_mk(vf, X, Y, n=10):
    BW=np.zeros(vf.shape)
    BW[Y.astype(int),X.astype(int)]=1
    BW[0:n,:]=0;BW[:,0:n]=0
    BW[-n-1:,:]=0;BW[:,-n-1:]=0
    props=regionprops_table(BW.astype(int))
    F=np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]])
    F=np.array([[1,1,1],[1,1,1],[1,1,1]])
    F=np.array([[0,1,0],[1,1,1],[0,1,0]])
    BW2=morphology.binary_dilation(BW,F)
    #threshold=filters.threshold_otsu(BW2)
    #mask = BW2 > threshold
    BW2 = morphology.remove_small_objects(BW2, 50, connectivity=8)
    #BW2=BW2*mask
    BW3=morphology.binary_erosion(BW2,F)
    # Create Matrix
    ColNrs, RowNrs = np.where(BW3 > 0.5)
    return ColNrs, RowNrs

def get_deformation_components_mk(u, v, ColNrs, RowNrs, n=5, m=5):
    fx=np.vstack((np.ones((n,m)), np.zeros((1,m)), -np.ones((n,m))))/n/m # note: u --> along-track, so x --> along-track, so is 'y-axis'
    fy=np.column_stack((np.ones((m,n)), np.zeros((n,1)), -np.ones((m,n))))/n/m
    fm=np.ones(fx.shape)/m/n;

    # first remove mean everywhere
    vm=v*1.0#-sp.signal.convolve2d(v,fm,'same');
    um=u*1.0#-sp.signal.convolve2d(u,fm,'same');
    DVDX=(conv2_mk(vm,fx,'same'))#/res*1000;
    DVDY=(conv2_mk(vm,fy,'same'))#/res*1000;
    DUDX=(conv2_mk(um,fx,'same'))#/res*1000;
    DUDY=(conv2_mk(um,fy,'same'))#/res*1000;
    # only the identified locations
    F=np.zeros(DUDX.shape);
    l=0;
    for i in range(0,len(ColNrs)):
        F[ColNrs[i]-l:ColNrs[i]+l+1,RowNrs[i]-l:RowNrs[i]+l+1]=1;
    #F[0:n,:]=0;F[:,0:n]=0
    #F[-n-1:,:]=0;F[:,-n-1:]=0
    DUDY=DUDY*F;
    DUDX=DUDX*F;
    DVDY=DVDY*F;
    DVDX=DVDX*F;
    DI=DUDX+DVDY;
    SH=(DVDX**2+DUDY**2)**0.5;

    return F, DI, SH

def label_average(img, labels, xy_interpolation=True):
    avg = np.zeros_like(img)
    ygrd, xgrd = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    
    for l in np.unique(labels):
        gpi = labels == l
        if gpi[gpi].size < 10 or not xy_interpolation:
            avg[gpi] = img[gpi].mean()
            continue
        x_l0 = xgrd[gpi]
        y_l0 = ygrd[gpi]
        i_l0 = img[gpi]
        xx = np.vstack([np.ones_like(x_l0), x_l0, y_l0]).T
        bb = np.linalg.lstsq(xx, i_l0[None].T, rcond=None)[0]
        i_l0_r = np.dot(xx, bb)
        avg[gpi] = i_l0_r.flat        
    return avg

def clustering_filter(u3, v3, n_clusters=20, med_filt_size=5, xy_interpolation=True):
    #u3, v3 = [zoomout(i, stp) for i in [uf, vf]]
    features = np.vstack([i.flatten() for i in [u3, v3]]).T
    connectivity = img_to_graph(u3)
    n_clusters_tot = int(n_clusters/1000*u3.shape[0])
    print('n_clusters_tot', n_clusters_tot)
    ward = AgglomerativeClustering(n_clusters=n_clusters_tot, linkage="ward", connectivity=connectivity)
    ward.fit(features)
    labels = ward.labels_.reshape(u3.shape)
    labelsf = median_filter(labels, med_filt_size)
    u4, v4 = [label_average(a, labelsf, xy_interpolation=xy_interpolation) for a in [u3, v3]]
    return u4, v4

def multi_look(inp, stp):
    return median_filter(inp, stp)[::stp, ::stp]

def get_deformation(u, v, use_diff=True):
    if use_diff:
        dudy = np.diff(u, axis=0)[:, :-1]
        dudx = np.diff(u, axis=1)[:-1, :]
        dvdy = np.diff(v, axis=0)[:, :-1]
        dvdx = np.diff(v, axis=1)[:-1, :]
    else:
        dudy, dudx = np.gradient(u)
        dvdy, dvdx = np.gradient(v)
    div = dudx + dvdy
    she = np.hypot(dudx - dvdy, dudy + dvdx)
    tot = np.hypot(div, she)
    return div, she, tot

def get_chunks(landmask, icemask, min_size=500):
    mask = minimum_filter(icemask.data, 10)
    mask[landmask] = True
    mask_azi_vec = mask.max(axis=1).astype(int)
    ch_starts = np.where(np.diff(mask_azi_vec) < 0)[0]+1
    ch_stops = np.where(np.diff(mask_azi_vec) > 0)[0]
    
    if mask_azi_vec[0] == 0:
        ch_starts = np.hstack([0, ch_starts])
    if mask_azi_vec[-1] == 0:
        ch_stops = np.hstack([ch_stops, mask_azi_vec.size-1])
    
    ch_sizes = ch_stops - ch_starts
    return ch_starts[ch_sizes > min_size], ch_stops[ch_sizes > min_size], ch_sizes[ch_sizes > min_size]

def get_projected_swath(ifile):
    xgrd = GLOBAL_DATA['xgrd']
    ygrd = GLOBAL_DATA['ygrd']
    ename = GLOBAL_DATA['ename']
    
    ds0 = np.load(ifile)
    x = ds0['x']
    y = ds0['y']
    
    defor_ifile = ifile.replace('.npz', '_defor.npz')        
    ds1 = np.load(defor_ifile)
    ti = ds1[f'{ename}i']
    tm = ds1[f'{ename}m']
    tc = ds1[f'{ename}c']
    stp = ds1['stp']
    x = x[::stp, ::stp]
    y = y[::stp, ::stp]
    tm = tm[::stp, ::stp]
    
    # shrink size to the smallest
    min_shape = [min(i.shape[j] for i in [x, ti, tm, tc]) for j in [0,1]]
    x, y, ti, tm, tc  = [i[:min_shape[0], :min_shape[1]] for i in [x, y, ti, tm, tc]]

    tree = cKDTree(np.vstack([x.flatten(), y.flatten()]).T)
    d, inds = tree.query(np.vstack([xgrd.flatten(), ygrd.flatten()]).T, k = 4)
    w = 1.0 / d**2

    wsum = np.sum(w, axis=1)
    tidw = []
    for t in [ti, tm, tc]:
        tt = np.sum(w * t.flatten()[inds], axis=1) / wsum
        tt[d[:,0]>10000] = np.nan
        tt.shape = xgrd.shape
        tidw.append(tt)
    return tidw

def get_deformation_mosaic(ifiles, xlim, ylim, cores=10, ename='tot'):
    global GLOBAL_DATA
    GLOBAL_DATA = {}
    GLOBAL_DATA['xgrd'], GLOBAL_DATA['ygrd'] = np.meshgrid(np.arange(*xlim), np.arange(*ylim))
    GLOBAL_DATA['ename'] = ename

    with Pool(cores) as p:
        tot_pro_all = p.map(get_projected_swath, ifiles)

    tot_pro_avg = []
    for i in range(len(tot_pro_all[0])):
        tot_pro = np.nanmean(np.dstack([t[i] for t in tot_pro_all]), axis=2)
        tot_pro_avg.append(tot_pro)
    return tot_pro_avg

def make_three_maps(pro_avg, xlim, ylim, cmap, clim):
    srs_dst = ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=60)
    map_extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    idx = [0,1, 2]

    fig, axs = plt.subplots(1,3, figsize=(15,5), subplot_kw={'projection': srs_dst})
    for i in range(3):
        imsh=axs[i].imshow(pro_avg[i], extent=[xlim[0], xlim[1], ylim[1], ylim[0]], cmap=cmap, clim=clim)
        axs[i].add_feature(LAND)
        axs[i].add_feature(COASTLINE)
        axs[i].set_extent(map_extent, crs=srs_dst)
        axs[i].plot(0, 0, 'bo')
        axs[i].text(30000, -30000, 'North \nPole')
    plt.tight_layout()
    plt.show()
