import numpy as np
from scipy.ndimage import zoom, median_filter, gaussian_filter, distance_transform_edt, minimum_filter
from scipy.interpolate import griddata
from matplotlib.tri import Triangulation, LinearTriInterpolator
from stereoid.sea_ice import RadarModel

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