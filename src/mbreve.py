######################################################################
# Author: Rohan Dahale, Date: 12 July 2024
######################################################################

# Import libraries
import numpy as np
import pandas as pd
import ehtim as eh
import ehtim.scattering.stochastic_optics as so
from preimcal import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb

import argparse
import os
import glob
from tqdm import tqdm
import itertools 
import sys
from copy import copy
from utilities import *
colors, titles, labels, mfcs, mss = common()

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, 
                   default='hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_dcal.uvfits', 
                   help='string of uvfits to data to compute chi2')
    p.add_argument('--kinemv', type=str, default='none', help='path of kine .hdf5')
    p.add_argument('--ehtmv',  type=str, default='none', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',  type=str, default='none', help='path of doghit .hdf5')
    p.add_argument('--ngmv',   type=str, default='none', help='path of ngmem .hdf5')
    p.add_argument('--resmv',  type=str, default='none', help='path of resolve .hdf5')
    p.add_argument('--modelingmv',  type=str, default='none', help='path of modeling .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./amp.png', 
                   help='name of output file with path')
    p.add_argument('--scat', type=str, default='none', help='onsky, deblur, dsct, none')

    return p

# List of parsed arguments
args = create_parser().parse_args()

outpath = args.outpath

paths={}

if args.modelingmv!='none':
    paths['modeling']=args.modelingmv
if args.ngmv!='none':
    paths['ngmem']=args.ngmv
if args.dogmv!='none':
    paths['doghit']=args.dogmv 
if args.ehtmv!='none':
    paths['ehtim']=args.ehtmv
if args.resmv!='none':
    paths['resolve']=args.resmv
if args.kinemv!='none':
    paths['kine']=args.kinemv
######################################################################

obs = eh.obsdata.load_uvfits(args.data)
obs = obs.avg_coherent(60)
obs = obs.flag_UT_range(UT_start_hour=10.89, UT_stop_hour=14.05, output='flagged')

obs.add_scans()
obs = obs.switch_polrep(polrep_out ='circ')
amp = pd.DataFrame(obs.data)
obs = obs.switch_polrep(polrep_out ='stokes')
times = []
for t in obs.scans:
    times.append(t[0])
obslist = obs.split_obs()    

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
######################################################################

polpaths={}
for p in paths.keys():
    mv=eh.movie.load_hdf5(paths[p])
    im=mv.get_image(times[0])
    if len(im.ivec)>0 and len(im.qvec)>0 and len(im.uvec)>0:
        polpaths[p]=paths[p]
    else:
        print('There is no I,Q or U')
######################################################################

mb_time, mb_window = dict(), dict()

print(polpaths.keys())

for p in polpaths.keys():
    mv = eh.movie.load_hdf5(polpaths[p])
    mb_time[p], mb_window[p] = [], []
    for ii in range(len(times)):
        tstamp = times[ii]
        im = mv.get_image(times[ii])
        im.rf = obslist_t[ii].rf
        im.ra=obslist_t[ii].ra
        im.dec=obslist_t[ii].dec
        if im.xdim%2 == 1:
            im = im.regrid_image(targetfov=im.fovx(), npix=im.xdim-1)
            im.rf=obslist_t[ii].rf
            im.ra=obslist_t[ii].ra
            im.dec=obslist_t[ii].dec
        obs_mod = im.observe_same(obslist_t[ii], add_th_noise=False, ttype='fast')
        obs_mod = obs_mod.switch_polrep(polrep_out ='circ')
        amp_mod = pd.DataFrame(obs_mod.data)
        # select baseline
        subtab  = select_baseline(amp_mod, 'AA', 'AZ')
        try:
            idx = np.where(np.round(subtab['time'].values,3)  == np.round(tstamp,3))[0][0]             
            mb_time[p].append(subtab['time'][idx]) 
            mb_window[p].append(abs(2*subtab['rlvis'][idx]/(subtab['rrvis'][idx]+subtab['rrvis'][idx])))  
        except:
            pass
    
######################################################################

stab  = select_baseline(amp, 'AA', 'AZ')
#mbreve = np.sqrt(abs(stab['qvis'])**2+abs(stab['uvis'])**2)/abs(stab['vis'])
mbreve = abs(2*stab['rlvis']/(stab['rrvis']+stab['rrvis']))
#mbreve_sig = np.sqrt((mbreve**2)*(stab['qsigma']**2-stab['usigma']**2+(stab['sigma']**2/abs(stab['vis'])**2)))/abs(stab['vis'])
x = np.abs(stab['rlvis'])
y = np.abs(stab['rrvis'])
z = np.abs(stab['llvis'])
#mbreve_sig = (1 / x) * np.sqrt(x**2 * ((stab['rlsigma']**2 / (y + z)**2) - ((stab['rrsigma']**2 + stab['llsigma']**2) / (y + z)**4)))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,6))
 #yerr=mbreve_sig,

ax.plot(stab['time'], mbreve, c='black', mfc='none', mec='black', marker='o', ls="-", lw=0.5, ms=10, alpha=1.0, label='AA-AZ')

for p in polpaths.keys():
    ax.plot(mb_time[p], mb_window[p], c=colors[p], marker='o', ms=5, ls="-", lw=0.5, label=labels[p], alpha=1.0)

ax.set_yscale('log')
ax.set_ylim(0.01,1)
ax.set_xlabel('Time (UTC)')
#plt.ylabel("$|\\breve{m}| \\approx \sqrt{|\\tilde{Q}|^2+|\\tilde{U}|^2}/|\\tilde{I}|$")
ax.set_ylabel("$|\\breve{m}| = |2RL/(RR+LL)|$")
ax.legend(ncols=3, loc='best',  bbox_to_anchor=(1.0, 1.2), markerscale=2.0, fontsize=16)
plt.savefig(args.outpath, bbox_inches='tight', dpi=300)