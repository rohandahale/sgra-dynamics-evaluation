######################################################################
# Author: Kotaro Moriyama, Date: 25 Mar 2024
######################################################################

# Import libraries
import numpy as np
import pandas as pd
import ehtim as eh
import ehtim.scattering.stochastic_optics as so
from preimcal import *
from ehtim.const_def import *
from tqdm import tqdm
from scipy import interpolate, optimize, stats
from scipy.interpolate import RectBivariateSpline
from copy import copy
from astropy.constants import k_B,c
import astropy.units as u

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
import itertools

import argparse
import os
import glob

from utilities import *
colors, titles, labels, mfcs, mss = common()

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str,
                   default='hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_dcal.uvfits',
                   help='string of uvfits to data to compute chi2')
    p.add_argument('--truthmv', type=str, default='none', help='path of truth .hdf5')
    p.add_argument('--kinemv',  type=str, default='none', help='path of kine .hdf5')
    p.add_argument('--ehtmv',   type=str, default='none', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',   type=str, default='none', help='path of doghit .hdf5')
    p.add_argument('--ngmv',    type=str, default='none', help='path of ngmem .hdf5')
    p.add_argument('--resmv',   type=str, default='none', help='path of resolve .hdf5')
    p.add_argument('--modelingmv',  type=str, default='none', help='path of modeling .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./chi2.png',
                   help='name of output file with path')
    p.add_argument('--scat', type=str, default='none', help='onsky, deblur, dsct, none')

    return p

# List of parsed arguments
args = create_parser().parse_args()

pathmovt = args.truthmv
outpath = args.outpath

paths={}
if args.truthmv!='none':
    paths['truth']=args.truthmv
if args.kinemv!='none':
    paths['kine']=args.kinemv
if args.resmv!='none':
    paths['resolve']=args.resmv
if args.ehtmv!='none':
    paths['ehtim']=args.ehtmv
if args.dogmv!='none':
    paths['doghit']=args.dogmv
if args.ngmv!='none':
    paths['ngmem']=args.ngmv
if args.modelingmv!='none':
    paths['modeling']=args.modelingmv
    
######################################################################


obs = eh.obsdata.load_uvfits(args.data)
obs, times, obslist_t, polpaths = process_obs(obs, args, paths)

outpath_csv={}

######################################################################

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(32,8), sharex=True)

ax[0,0].set_ylabel('Diameter $({\mu as})$')
ax[0,1].set_ylabel('FWHM $({\mu as})$')
ax[0,2].set_ylabel('Position angle ($^\circ$)')

ax[1,0].set_ylabel('Frac. Central Brightness')
ax[1,1].set_ylabel('Asymmetry')
ax[1,2].set_ylabel('Peak PA ($^\circ$)')

ax[1,0].set_xlabel('Time (UTC)')
ax[1,1].set_xlabel('Time (UTC)')
ax[1,2].set_xlabel('Time (UTC)')


polpaths={}
for p in paths.keys():
    mv=eh.movie.load_hdf5(paths[p])
    im=mv.im_list()[0]
    if len(im.ivec)>0 and len(im.qvec)>0 and len(im.uvec)>0 :
        polpaths[p]=paths[p]

polvpaths={}
for p in paths.keys():
    mv=eh.movie.load_hdf5(paths[p])
    im=mv.im_list()[0]
    if len(im.ivec)>0 and len(im.vvec)>0:
        polvpaths[p]=paths[p]

for p in paths.keys():
    outpath_csv[p]= outpath+f'_{p}.csv'
    mv=eh.movie.load_hdf5(paths[p])

    imlist = [mv.get_image(t) for t in times]

    mv_ave = mv.avg_frame()
    # find ring center with the averaged image
    xc,yc = fit_ring(mv_ave)
    ring_outputs = [extract_ring_quantites(im_f,xc=xc,yc=yc) for im_f in tqdm(imlist)]
    table = pd.DataFrame(ring_outputs, columns=["time", "D","Derr","W","Werr", "true_D", "true_Derr", "PAori","PAerr","papeak","A","Aerr","fc","xc","yc","fwhm_maj","fwhm_min","hole_flux","outer_flux","ring_flux","totalflux","hole_dflux","outer_dflux","ring_dflux"])
    #table_vals[p]=np.round(np.mean(np.array(mnet_tab)),3)
    
    table.to_csv(outpath_csv[p], index=False)

    mc=colors[p]
    alpha = 1.0
    lc=colors[p]
    mfc=mfcs[p]
    ms=mss[p]
    # Diameter
    ax[0,0].plot(times, table["D"],  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha, label=labels[p])
    # FWHM
    ax[0,1].plot(times, table["W"],  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)
    # Position angle
    ax[0,2].plot(times, table["PAori"],  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)
    # Frac Central Brightness
    ax[1,0].plot(times, table["fc"],  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)
    # Asymetry
    ax[1,1].plot(times, table["A"],  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)
    # peak position angle
    ax[1,2].plot(times, table["papeak"],  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)

ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(3., 1.3), markerscale=2.0)
plt.savefig(args.outpath+'.png', bbox_inches='tight', dpi=300)


######################################################################
# Stokes QUV
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(32,8), sharex=True)
#"mnet", "mavg", "evpa", "beta2_abs", "beta2_angle", "vnet"
ax[0,0].set_ylabel('$|m|_{net}$')
ax[0,1].set_ylabel(r'$\langle |m|\rangle$')
ax[0,2].set_ylabel('$\chi\ (^\circ)$')

ax[1,0].set_ylabel(r'$\beta_2$')
ax[1,1].set_ylabel(r'$\angle \beta_2\ (^\circ)$')
ax[1,2].set_ylabel('$|v|_{net}$')

ax[1,0].set_xlabel('Time (UTC)')
ax[1,1].set_xlabel('Time (UTC)')
ax[1,2].set_xlabel('Time (UTC)')

#ax[0].set_ylim(0.0,0.1)
#ax[1].set_ylim(0.0,0.15)
#ax[2].set_ylim(0.0,0.05)
#ax[3].set_ylim(0.0,0.1)


pol_count = 0
for p in polpaths.keys():
    mv=eh.movie.load_hdf5(polpaths[p])

    imlist = [mv.get_image(t) for t in times]

    if len(imlist[0].qvec)==len(imlist[0].ivec):
        pol_count += 1
        mv_ave = mv.avg_frame()
        # find ring center with the averaged image
        xc,yc = fit_ring(mv_ave)
        pol_outputs = [extract_pol_quantites(im_f,xc=xc,yc=yc) for im_f in tqdm(imlist)]

        # if kine like shifting center, need use the following
        # pol_outputs = [extract_pol_quantites(image_f) for image_f in tqdm.tqdm(ims_rg[0].im_list())]

        table = pd.DataFrame(pol_outputs, columns=["time_utc","mnet","mavg","evpa","beta2_abs","beta2_angle","vnet","beta_v_abs", "beta_v_angle","beta2_v_abs", "beta2_v_angle"])

        mc=colors[p]
        alpha = 1.0
        lc=colors[p]
        mfc=mfcs[p]
        ms=mss[p]
        # Diameter
        ax[0,0].plot(times, table["mnet"],  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha, label=labels[p])
        # FWHM
        ax[0,1].plot(times, table["mavg"],  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)
        # Position angle
        ax[0,2].plot(times, table["evpa"],  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)
        # Frac Central Brightness
        ax[1,0].plot(times, table["beta2_abs"],  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)
        # Asymetry
        ax[1,1].plot(times, table["beta2_angle"],  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)
        # peak position angle
        ax[1,2].plot(times, table["vnet"],  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)

if pol_count>0:
    ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(3., 1.3), markerscale=2.0)
    plt.savefig(args.outpath+'_pol.png', bbox_inches='tight', dpi=300)
