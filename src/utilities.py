######################################################################
# Author: Rohan Dahale, Date: 12 July 2024
######################################################################

#Import libraries
import numpy as np
import pandas as pd
import ehtim as eh
import ehtim.scattering.stochastic_optics as so
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
import argparse
import os
import glob
from tqdm import tqdm
import itertools 
import sys
import scipy
from copy import copy
from copy import deepcopy
import matplotlib as mpl

#Rex
from ehtim.const_def import *
from scipy import interpolate, optimize, stats
from scipy.interpolate import RectBivariateSpline
from astropy.constants import k_B,c
import astropy.units as u
#Rex

from preimcal import *


def common():
    ######################################################################
    # Plotting Setup
    ######################################################################
    import matplotlib as mpl
    mpl.rcParams['figure.dpi']=300
    plt.rcParams["xtick.direction"]="in"
    plt.rcParams["ytick.direction"]="in"
    
    mpl.rcParams["axes.labelsize"] = 20
    mpl.rcParams["xtick.labelsize"] = 18
    mpl.rcParams["ytick.labelsize"] = 18
    mpl.rcParams["legend.fontsize"] = 18
    
    from matplotlib import font_manager
    font_dirs = font_manager.findSystemFonts(fontpaths='./fonts/', fontext="ttf")
    
    fe = font_manager.FontEntry(
        fname='./fonts/Helvetica.ttf',
        name='Helvetica')
    font_manager.fontManager.ttflist.insert(0, fe) # or append is fine
    mpl.rcParams['font.family'] = fe.name # = 'your custom ttf font name'
    ######################################################################
    
    colors = {
            'truth'    : 'black',
            'kine'     : 'tab:blue',
            'resolve'  : 'tab:orange',
            'ehtim'    : 'tab:green',
            'doghit'   : 'tab:red',
            'ngmem'    : 'tab:purple',
            'modeling' : 'tab:brown' 
        }
    
    titles = {  
            'truth'      : 'Truth',
            'kine'       : 'kine',
            'resolve'    : 'resolve',
            'ehtim'      : 'ehtim',
            'doghit'     : 'doghit',
            'ngmem'      : 'ngmem',
            'modeling'   : 'modeling'
        }
    
    labels = {  
            'truth'      : 'Truth',
            'kine'       : 'kine',
            'resolve'    : 'resolve',
            'ehtim'      : 'ehtim',
            'doghit'     : 'doghit',
            'ngmem'      : 'ngmem',
            'modeling'   : 'modeling'
        }

    mfcs = {
            'truth'    : 'none',
            'kine'     : 'tab:blue',
            'resolve'  : 'tab:orange',
            'ehtim'    : 'tab:green',
            'doghit'   : 'tab:red',
            'ngmem'    : 'tab:purple',
            'modeling' : 'tab:brown' 
        }

    mss = {
                'truth'    : 10,
                'kine'     : 5,
                'resolve'  : 5,
                'ehtim'    : 5,
                'doghit'   : 5,
                'ngmem'    : 5,
                'modeling' : 5 
            }

    return colors, titles, labels, mfcs, mss
    
def process_obs(obs,args,paths):
    obs.add_scans()
    obs = obs.avg_coherent(60)
    #obs = obs.flag_UT_range(UT_start_hour=10.89, UT_stop_hour=14.05, output='flagged')
    obs.add_scans()
    obslist = obs.split_obs()
    times = []
    for o in obslist:
        times.append(o.data['time'][0])
    
    # Truncating the times and obslist based on submitted movies
    obslist_tn=[]
    min_arr=[] 
    max_arr=[]
    for p in paths.keys():
        mv=eh.movie.load_hdf5(paths[p])
        min_arr.append(min(mv.times))
        max_arr.append(max(mv.times))
    x=np.argwhere(times>max(min_arr))
    ntimes=[]
    for t in x:
        ntimes.append(times[t[0]])
        obslist_tn.append(obslist[t[0]])
    times=[]
    obslist_t=[]
    y=np.argwhere(min(max_arr)>ntimes)
    for t in y:
        times.append(ntimes[t[0]])
        obslist_t.append(obslist_tn[t[0]])
        
    if hasattr(args, 'pol'):
        pol = args.pol
        polpaths={}
        for p in paths.keys():
            mv=eh.movie.load_hdf5(paths[p])
            im=mv.get_image(times[0])

            if pol=='I':
                if len(im.ivec)>0:
                    polpaths[p]=paths[p]
                else:
                    print(f'{p}: Parse a vaild I pol value')
            elif pol=='Q':
                if len(im.qvec)>0:
                    polpaths[p]=paths[p]
                else:
                    print(f'{p}: Parse a vaild Q pol value')
            elif pol=='U':
                if len(im.uvec)>0:
                    polpaths[p]=paths[p]
                else:
                    print(f'{p}: Parse a vaild U pol value')
            elif pol=='V':
                if len(im.vvec)>0:
                    polpaths[p]=paths[p]
                else:
                    print(f'{p}: Parse a vaild V pol value')
            else:
                print('Parse a vaild pol value')
    else:
        polpaths=None
        
    return obs, times, obslist_t, polpaths


def select_baseline(tab, st1, st2):
    stalist = list(itertools.permutations([st1, st2]))
    idx = []
    for stations in stalist:
        ant1, ant2 = stations
        subidx = np.where((tab["t1"].values == ant1) &
                          (tab["t2"].values == ant2) )
        idx +=  list(subidx[0])

    newtab = tab.take(idx).sort_values(by=["time"]).reset_index(drop=True)
    return newtab


def compute_ramesh_metric(us, vs, N=None):

    if N is None:
        N = len(us)

    mean_u2 = np.sum([u**2. for u in us]) / (2.*N)
    mean_v2 = np.sum([v**2. for v in vs]) / (2.*N)
    mean_uv = np.sum([us[i]*vs[i] for i in range(len(us))]) / (2.*N)

    numerator = np.sqrt( (mean_u2-mean_v2)**2. + 4*(mean_uv**2.) )
    denominator = mean_u2 + mean_v2

    return 1 - numerator / denominator

def select_triangle(tab, st1, st2, st3):
    stalist = list(itertools.permutations([st1, st2, st3]))
    idx = []
    for stations in stalist:
        ant1, ant2, ant3 = stations
        subidx = np.where((tab["t1"].values == ant1) &
                          (tab["t2"].values == ant2) &
                          (tab["t3"].values == ant3) )
        idx +=  list(subidx[0])

    newtab = tab.take(idx).sort_values(by=["time"]).reset_index(drop=True)
    return newtab

# reduced-chi2 of closure phases
def rchi_cp(cph, cph_sigma, cph_mod):
    ''' reduced-chi2 of closure phases '''
    rchicp = np.sum( (1 - np.cos(np.deg2rad(cph-cph_mod)))/(np.deg2rad(cph_sigma)**2) ) * 2/len(cph)
    return rchicp

def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def radial_homogeneity(u, v):

    uvdists = np.sqrt(u**2. + v**2.)

    uvdists = sorted(uvdists)
    uvdists = list(uvdists)
    uvdists.append(1e10)
    uvdists.append(0e10)
    uvdists = np.array(uvdists)
    score = jensen_shannon_distance(uvdists, [i*1.e10/len(uvdists) for i in range(len(uvdists))])
    
    return score



def isotropy_metric_normalized(u, v, i_max=None, r_max=None):

    if i_max is None or r_max is None: print("Please provide a value for both i_max and r_max.")

    ## here we're just following the formula in the paper
    iso = compute_ramesh_metric(u, v)
    rad_hom = radial_homogeneity(u, v)

    return (iso/i_max) * (1 - rad_hom/r_max)


def get_nxcorr_cri_beam(im, beamparams, pol):
    im_blur = im.blur_gauss(beamparams, frac_pol=1.0)
    nx = im.compare_images(im_blur, pol=pol)[0][0]
    return nx

def process_obs_weights(obs,args,paths):
    obs.add_scans()
    obs = obs.avg_coherent(60)
    obs.add_scans()
    obslist = obs.split_obs()
    times = []
    for o in obslist:
        times.append(o.data['time'][0])    

    ######################################################################

    # Truncating the times and obslist based on submitted movies
    obslist_tn=[]
    min_arr=[] 
    max_arr=[]
    for p in paths.keys():
        mv=eh.movie.load_hdf5(paths[p])
        min_arr.append(min(mv.times))
        max_arr.append(max(mv.times))
    x=np.argwhere(times>max(min_arr))
    ntimes=[]
    for t in x:
        ntimes.append(times[t[0]])
        obslist_tn.append(obslist[t[0]])
    times=[]
    obslist_t=[]
    y=np.argwhere(min(max_arr)>ntimes)
    for t in y:
        times.append(ntimes[t[0]])
        obslist_t.append(obslist_tn[t[0]])

    obs_t=eh.obsdata.merge_obs(obslist_t)
    obs_t.add_scans()
    splitObs = obs_t.split_obs()

    ######################################################################

    i_s, r_s = [], []
    for j, s_obs in enumerate(splitObs):
        if 'AA' in s_obs.data['t1'] or 'AA' in s_obs.data['t2']:
            s_obs = s_obs.flag_sites('AP')
        if 'SM' in s_obs.data['t1'] or 'SM' in s_obs.data['t2']: 
                s_obs = s_obs.flag_sites('JC')

        if len(s_obs.data)==0: continue

        unpackedobj = np.transpose(s_obs.unpack(['u', 'v'], debias=True, conj=True))
        u = unpackedobj['u']
        v = unpackedobj['v']
        i_s.append(compute_ramesh_metric(u, v))
        r_s.append(radial_homogeneity(u, v))

    imax = 1 #max(i_s) #1 # convention choice, can also be max(i_s)
    rmax = 0.513 #max(r_s)

    I=[]
    snr={}
    snr['I']=[]
    snr['Q']=[]
    snr['U']=[]
    snr['V']=[]

    for j, s_obs in enumerate(splitObs):
        ## here we're flagging a zero baseline as an example if you choose; it will
        ## just scale the overall score since N will change
        if 'AA' in s_obs.data['t1'] or 'AA' in s_obs.data['t2']: 
            s_obs = s_obs.flag_sites('AP')
        if 'JC' in s_obs.data['t1'] or 'JC' in s_obs.data['t2']: 
            s_obs = s_obs.flag_sites('SM')
        if len(s_obs.data)==0: continue
        unpackedobj = np.transpose(s_obs.unpack(['u', 'v'], debias=True, conj=True))
        u = unpackedobj['u']
        v = unpackedobj['v']
        I.append(isotropy_metric_normalized(u, v, i_max=imax, r_max=rmax))

    I=np.array(I)
    I=I/np.max(I)
    
    for i in range(len(obslist_t)):
        df=pd.DataFrame(obslist_t[i].data)
        snr['I'].append(np.mean(np.abs(df['vis'])/df['sigma']))
        snr['Q'].append(np.mean(np.abs(df['qvis'])/df['qsigma']))
        snr['U'].append(np.mean(np.abs(df['uvis'])/df['usigma']))
        snr['V'].append(np.mean(np.abs(df['vvis'])/df['vsigma']))

    for x in snr.keys():    
        snr[x] = np.array(snr[x])
        # Squeeze SNR between min and max of I
        snr[x] = np.min(I) + ((snr[x] - np.min(snr[x])) / (np.max(snr[x]) - np.min(snr[x]))) * (np.max(I) - np.min(I))
        
    w_norm={}
    w_norm['I']=[]
    w_norm['Q']=[]
    w_norm['U']=[]
    w_norm['V']=[]

    for x in w_norm.keys():
        w = I*snr[x]
        w_sum=np.sum(w)
        w_norm[x]=np.array(w/w_sum)
    
    return obs, obs_t, obslist_t, splitObs, times, I, snr, w_norm



######################################################################
# BEGIN: Author: Kotaro Moriyama, Date: 25 Mar 2024
######################################################################

######################################################################
# REx functions
######################################################################

def calculate_true_d_error(D, W, D_err, W_err):
    """
    Calculates the propagated error for the quantity true_D.

    Args:
        D (float or np.ndarray): The measured value(s) of D.
        W (float or np.ndarray): The measured value(s) of W.
        D_err (float or np.ndarray): The error in the measurement(s) of D.
        W_err (float or np.ndarray): The error in the measurement(s) of W.

    Returns:
        float or np.ndarray: The propagated error in true_D.
    """
    # For clarity, let's define some intermediate terms
    ln2 = np.log(2)
    ratio = W / D
    ratio_sq = ratio**2
    
    # Common denominator term in the partial derivatives
    common_denominator = (1 - (1 / (4 * ln2)) * ratio_sq)**2

    # Partial derivative of true_D with respect to D
    partial_d = (1 - (3 / (4 * ln2)) * ratio_sq) / common_denominator

    # Partial derivative of true_D with respect to W
    partial_w = ((1 / (2 * ln2)) * ratio) / common_denominator

    # Calculate the squared error terms
    d_err_term_sq = np.square(partial_d * D_err)
    w_err_term_sq = np.square(partial_w * W_err)

    # The final propagated error is the square root of the sum of squared terms
    true_d_err = np.sqrt(d_err_term_sq + w_err_term_sq)

    return true_d_err

def extract_ring_quantites(image,xc=None,yc=None, rcutoff=5):
    Npa=360
    Nr=100

    if xc==None or yc==None:
    # Compute the image center -----------------------------------------------------------------------------------------
        xc,yc = fit_ring(image)
    # Gridding and interpolation ---------------------------------------------------------------------------------------
    x= np.arange(image.xdim)*image.psize/RADPERUAS
    y= np.flip(np.arange(image.ydim)*image.psize/RADPERUAS)
    z = image.imarr()
    f_image = interpolate.interp2d(x,y,z,kind="cubic") # init interpolator
    #f_image = RectBivariateSpline(x, y, z)

    # Create a mesh grid in polar coordinates
    radial_imarr = np.zeros([Nr,Npa])

    pa = np.linspace(0,360,Npa)
    pa_rad = np.deg2rad(pa)
    radial = np.linspace(0,50,Nr)
    dr = radial[-1]-radial[-2]

    Rmesh, PAradmesh = np.meshgrid(radial, pa_rad)
    x = Rmesh*np.sin(PAradmesh) + xc
    y = Rmesh*np.cos(PAradmesh) + yc
    for r in range(Nr):
        z = [f_image(x[i][r],y[i][r]) for i in range(len(pa))]
        radial_imarr[r,:] = np.array(z)[:,0]
    radial_imarr = np.fliplr(radial_imarr)
    # Calculate the r_pk at each PA and average -> using radius  --------------------------------------------------------
    # Caluculating the ring width from rmin and rmax
    peakpos = np.unravel_index(np.argmax(radial_imarr), shape=radial_imarr.shape)

    Rpeak=[]
    Rmin=[]
    Rmax=[]
    ridx_r50= np.argmin(np.abs(radial - 50))
    I_floor = radial_imarr[ridx_r50,:].mean()
    for ipa in range(len(pa)):
        tmpIr = copy(radial_imarr[:,ipa])-I_floor
        tmpIr[np.where(radial < rcutoff)]=0
        ridx_pk = np.argmax(tmpIr)
        rpeak = radial[ridx_pk]
        if ridx_pk > 0 and ridx_pk < Nr-1:
            val_list= tmpIr[ridx_pk-1:ridx_pk+2]
            rpeak = quad_interp_radius(rpeak, dr, val_list)[0]
        idx = np.array(np.where(tmpIr > tmpIr.max()/2.0))
        Rpeak.append(rpeak)
        # if tmpIr < 0, make rmin & rmax nan
        rmin,rmax = calc_width(tmpIr,radial,rpeak)
        # append
        Rmin.append(rmin)
        Rmax.append(rmax)
    paprofile = pd.DataFrame()
    paprofile["PA"] = pa
    paprofile["rpeak"] = Rpeak
    paprofile["rhalf_max"]=Rmax
    paprofile["rhalf_min"]=Rmin

    D = np.mean(paprofile["rpeak"]) * 2
    Derr = paprofile["rpeak"].std() * 2
    W = np.mean(paprofile["rhalf_max"] - paprofile["rhalf_min"])
    Werr =  (paprofile["rhalf_max"] - paprofile["rhalf_min"]).std()

    # Caluculate the orienttion angle, contrast, and assymetry
    rin  = D/2.-W/2.
    rout  = D/2.+W/2.
    if rin <= 0.:
        rin  = 0.

    exptheta =np.exp(1j*pa_rad)

    pa_ori_r=[]
    amp_r = []
    ridx1 = np.argmin(np.abs(radial - rin))
    ridx2 = np.argmin(np.abs(radial - rout))
    for r in range(ridx1, ridx2+1, 1):
        amp =  (radial_imarr[r,:]*exptheta).sum()/(radial_imarr[r,:]).sum()
        amp_r.append(amp)
        pa_ori = np.angle(amp, deg=True)
        pa_ori_r.append(pa_ori)
    pa_ori_r=np.array(pa_ori_r)
    amp_r = np.array(amp_r)
    PAori = stats.circmean(pa_ori_r,high=360,low=0)
    PAerr = stats.circstd(pa_ori_r,high=360,low=0)
    A = np.mean(np.abs(amp_r))
    Aerr = np.std(np.abs(amp_r))

    ridx_r5= np.argmin(np.abs(radial - 5))
    ridx_pk = np.argmin(np.abs(radial - D/2))
    fc = radial_imarr[0:ridx_r5,:].mean()/radial_imarr[ridx_pk,:].mean()

    # source size from 2nd moment
    fwhm_maj,fwhm_min,theta = image.fit_gauss()
    fwhm_maj /= RADPERUAS
    fwhm_min /= RADPERUAS


    # calculate flux ratio
    Nxc = int(xc/image.psize*RADPERUAS)
    Nyc = int(yc/image.psize*RADPERUAS)
    hole = extract_hole(image,Nxc,Nyc,r=rin)
    ring = extract_ring(image,Nxc,Nyc,rin=rin, rout=rout)
    outer = extract_outer(image,Nxc,Nyc,r=rout)
    hole_flux = hole.total_flux()
    outer_flux = outer.total_flux()
    ring_flux = ring.total_flux()

    Shole  = np.pi*rin**2
    Souter = (2.*rout)**2.-np.pi*rout**2
    Sring = np.pi*rout**2-np.pi*rin**2

    # convert uas^2 to rad^2
    Shole = Shole*RADPERUAS**2
    Souter = Souter*RADPERUAS**2
    Sring = Sring*RADPERUAS**2

    #unit K brighthness temperature
    freq = image.rf*u.Hz
    hole_dflux  = hole_flux/Shole*(c**2/2/k_B/freq**2).to(u.K/u.Jansky).value
    outer_dflux = outer_flux/Souter*(c**2/2/k_B/freq**2).to(u.K/u.Jansky).value
    ring_dflux = ring_flux/Sring*(c**2/2/k_B/freq**2).to(u.K/u.Jansky).value
    
    true_D=np.array(D/(1-(1/(4*np.log(2)))*(W/D)**2))
    true_Derr = calculate_true_d_error(D, W, Derr, Werr)

    # output dictionary
    outputs = dict(
        time = image.time,
        radial_imarr=radial_imarr,
        peak_idx=peakpos,
        rpeak=radial[peakpos[0]],
        papeak=pa[peakpos[1]],
        paprofile=paprofile,
        xc=xc,
        yc=yc,
        r = radial,
        PAori = PAori,
        PAerr = PAerr,
        A = A,
        Aerr = Aerr,
        fc = fc,
        D = D,
        Derr = Derr,
        W = W,
        Werr = Werr,
        true_D = true_D,
        true_Derr = true_Derr,
        fwhm_maj=fwhm_maj,
        fwhm_min=fwhm_min,
        hole_flux = hole_flux,
        outer_flux = outer_flux,
        ring_flux = ring_flux,
        totalflux = image.total_flux(),
        hole_dflux = hole_dflux,
        outer_dflux = outer_dflux,
        ring_dflux = ring_dflux
    )
    return outputs

# Clear ring structures
def extract_hole(image,Nxc,Nyc, r=30):
    outimage = deepcopy(image)
    x = (np.arange(outimage.xdim)-Nxc+1)*outimage.psize/RADPERUAS
    y =  (np.arange(outimage.ydim)-Nyc+1)*outimage.psize/RADPERUAS
    x,y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - r**2>=0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim*outimage.xdim)
    return outimage

def extract_outer(image,Nxc,Nyc, r=30):
    outimage = deepcopy(image)
    x = (np.arange(outimage.xdim)-Nxc+1)*outimage.psize/RADPERUAS
    y =  (np.arange(outimage.ydim)-Nyc+1)*outimage.psize/RADPERUAS
    x,y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - r**2<=0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim*outimage.xdim)
    return outimage

def extract_ring(image, Nxc,Nyc,rin=30,rout=50):
    outimage = deepcopy(image)
    x = (np.arange(outimage.xdim)-Nxc+1)*outimage.psize/RADPERUAS
    y =  (np.arange(outimage.ydim)-Nyc+1)*outimage.psize/RADPERUAS
    x,y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - rin**2<=0)] = 0
    masked[np.where(x**2 + y**2 - rout**2>=0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim*outimage.xdim)

    return outimage

def quad_interp_radius(r_max, dr, val_list):
    v_L = val_list[0]
    v_max = val_list[1]
    v_R = val_list[2]
    rpk = r_max + dr*(v_L - v_R) / (2 * (v_L + v_R - 2*v_max))
    vpk = 8*v_max*(v_L + v_R) - (v_L - v_R)**2 - 16*v_max**2
    vpk /= (8*(v_L + v_R - 2*v_max))
    return (rpk, vpk)

def calc_width(tmpIr,radial,rpeak):
    spline = interpolate.UnivariateSpline(radial, tmpIr-0.5*tmpIr.max(), s=0)
    roots = spline.roots()  # find the roots

    if len(roots) == 0:
        return(radial[0], radial[-1])

    rmin = radial[0]
    rmax = radial[-1]
    for root in np.sort(roots):
        if root < rpeak:
            rmin = root
        else:
            rmax = root
            break

    return (rmin, rmax)

def fit_ring(image,Nr=50,Npa=25,rmin_search = 10,rmax_search = 100,fov_search = 0.1,Nserch =20):
    # rmin_search,rmax_search must be diameter
    image_blur = image.blur_circ(2.0*RADPERUAS,fwhm_pol=0)
    image_mod = image_blur.threshold(cutoff=0.05)
    image_mod = image
    xc,yc = eh.features.rex.findCenter(image_mod, rmin_search=rmin_search, rmax_search=rmax_search,
                         nrays_search=Npa, nrs_search=Nr,
                         fov_search=fov_search, n_search=Nserch)
    return xc,yc

# polarization functions ##############################
def make_polar_imarr(imarr, dx, xc=None, yc=None, rmax=50, Nr=50, Npa=180, kind="linear", image=None):
    '''
    Image array with polar coordinates
    Args:
        imarr (np.ndarray): 1dimensional image array. shape=(ny, nx)
        dx (float): pixel size with the x, y axis
        xc,yc: center of the ring
        rmax (float): maximum radial coordinate for the polar coordinates
        Nr, Npa (int): pixel number of the polar coordinates
        kind (str): kind of interpolation for polar coordinates
    Return:
        radial_imarr (np.ndarray)
    '''
    nx,ny = imarr.shape
    dy=dx
    x= np.arange(nx)*dx/RADPERUAS
    y= np.arange(ny)*dy/RADPERUAS
    #xc, yc =(np.max(x)-np.min(x))/2, (np.max(y)-np.min(y))/2
    if xc==None or yc==None:
    # Compute the image center
        xc,yc = fit_ring(image)

    z = imarr
    f_image = interpolate.interp2d(x,y,z,kind=kind)

    # Create a mesh grid in polar coordinates
    radial_imarr = np.zeros([Nr,Npa])
    pa = np.linspace(0,360,Npa)
    pa_rad = np.deg2rad(pa)
    radius = np.linspace(0,rmax,Nr)
    dr = radius[-1]-radius[-2]

    # interpolation with polar coordinates
    Rmesh, PAradmesh = np.meshgrid(radius, pa_rad)
    x, y = Rmesh*np.sin(PAradmesh) + xc, Rmesh*np.cos(PAradmesh) + yc
    for ir in range(Nr):
        z = [f_image(x[ipa][ir],y[ipa][ir]) for ipa in range(Npa)]
        radial_imarr[ir,:] = z[:]
    radial_imarr = np.fliplr(radial_imarr)

    return radial_imarr,radius, pa

def extract_pol_quantites(im,xc=None, yc=None, blur_size=-1):
    '''
    Calculate polarization quantites with the input image object and blur size in the unit of uas
    Returns:
        net fractional linear polarization, image averaged linear polarization, EVPA (deg), beta2 magnitude and phase (deg), fractional circular polarization
    '''
    Itot, Qtot, Utot = sum(im.imvec), sum(im.qvec), sum(im.uvec)
    if len(im.vvec)==0:
        print("Caution: no stokes V")
        im.vvec = np.zeros_like(im.imvec)
    Vtot = sum(im.vvec)
    # net fractional linear polarization
    mnet=np.sqrt(Qtot*Qtot + Utot*Utot)/Itot

    # image averaged linear polarization
    if blur_size<0:
        mavg = sum(np.sqrt(im.qvec**2 + im.uvec**2))/Itot
    else:
        im_blur = im.blur_circ(blur_size*eh.RADPERUAS, fwhm_pol=blur_size*eh.RADPERUAS)
        mavg = sum(np.sqrt(im_blur.qvec**2 + im_blur.uvec**2))/np.sum(im_blur.imvec)

    # evpa
    evpa =  (180./np.pi)*0.5*np.angle(Qtot+1j*Utot)

    # fractional circular polarization
    vnet = np.abs(Vtot)/Itot

    # beta2
    P = im.qvec+ 1j*im.uvec
    P_radial, radius, pa = make_polar_imarr(P.reshape(im.xdim, im.xdim), dx=im.psize, xc=xc, yc=yc, image=im)
    I_radial, dummy, dummy = make_polar_imarr(im.imvec.reshape(im.xdim, im.xdim), dx=im.psize, xc=xc, yc=yc, image=im)
    V_radial, dummy, dummy = make_polar_imarr(im.vvec.reshape(im.xdim, im.xdim), dx=im.psize, xc=xc, yc=yc, image=im)
    Pring, Vring, Vring2, Iring = 0, 0, 0, 0
    for ir, ipa in itertools.product(range(len(radius)), range(len(pa))):
        Pring += P_radial[ir, ipa] * np.exp(-2*1j*np.deg2rad(pa[ipa])) * radius[ir]
        Vring2 += V_radial[ir, ipa] * np.exp(-2*1j*np.deg2rad(pa[ipa])) * radius[ir]
        Vring  += V_radial[ir, ipa] * np.exp(-1*1j*np.deg2rad(pa[ipa])) * radius[ir]
        Iring += I_radial[ir, ipa] * radius[ir]
    beta2 = Pring/Iring
    beta2_abs, beta2_angle = np.abs(beta2), np.rad2deg(np.angle(beta2))

    beta2_v = Vring2/Iring
    beta2_v_abs, beta2_v_angle = np.abs(beta2_v), np.rad2deg(np.angle(beta2_v))
    beta_v = Vring/Iring
    beta_v_abs, beta_v_angle = np.abs(beta_v), np.rad2deg(np.angle(beta_v))

    # output dictionary
    outputs = dict(
        time_utc = im.time,
        mnet = mnet,
        mavg = mavg,
        evpa = evpa,
        beta2_abs = beta2_abs,
        beta2_angle = beta2_angle,
        vnet = vnet,
        beta_v_abs = beta_v_abs,
        beta_v_angle = beta_v_angle,
        beta2_v_abs = beta2_v_abs,
        beta2_v_angle = beta2_v_angle
        )
    return outputs

# polarization functions ##############################

######################################################################
# END: Author: Kotaro Moriyama, Date: 25 Mar 2024
######################################################################

# Function to calculate normalized weighted RMSE score
def normalized_rmse(true_signal, measured_signal, weights):
    # Ensure weights are positive and normalized
    weights = weights / np.sum(weights)  # Normalize weights to sum to 1
    rmse = np.sqrt(np.sum(weights * (measured_signal - true_signal) ** 2))
    scaling_factor = np.mean(np.abs(true_signal)) if np.mean(np.abs(true_signal)) != 0 else 1  # Avoid division by zero
    norm_rmse = rmse / scaling_factor
    return  np.round(np.exp(-norm_rmse),2) # Ensure the score is between 0 and 1