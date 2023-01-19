import numpy as np
from scipy.ndimage import zoom, median_filter
from scipy.interpolate import griddata
from matplotlib.tri import Triangulation, LinearTriInterpolator

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