######################################################################
# Author: Ilje Cho, Rohan Dahale, Date: 14 May 2024
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
    p.add_argument('--resmv',  type=str, default='none',help='path of resolve .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./chi2.png', 
                   help='name of output file with path')
    p.add_argument('--pol',  type=str, default='I',help='I,Q,U,V')
    p.add_argument('--scat', type=str, default='none', help='sct, dsct, none')

    return p

# List of parsed arguments
args = create_parser().parse_args()
######################################################################
# Plotting Setup
######################################################################
#plt.rc('text', usetex=True)
import matplotlib as mpl
#mpl.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 'monospace': ['Computer Modern Typewriter']})
mpl.rcParams['figure.dpi']=300
#mpl.rcParams["mathtext.default"] = 'regular'
plt.rcParams["xtick.direction"]="in"
plt.rcParams["ytick.direction"]="in"
#plt.style.use('dark_background')
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 18
mpl.rcParams["ytick.labelsize"] = 18
mpl.rcParams["legend.fontsize"] = 18

from matplotlib import font_manager
font_dirs = font_manager.findSystemFonts(fontpaths='./fonts/', fontext="ttf")
#mpl.rc('text', usetex=True)

fe = font_manager.FontEntry(
    fname='./fonts/Helvetica.ttf',
    name='Helvetica')
font_manager.fontManager.ttflist.insert(0, fe) # or append is fine
mpl.rcParams['font.family'] = fe.name # = 'your custom ttf font name'
######################################################################

# Time average data to 60s
obs = eh.obsdata.load_uvfits(args.data)
obs.add_scans()
# From GYZ: If data used by pipelines is descattered (refractive + diffractive),
# Add 2% error and deblur original data.
if args.scat=='dsct':
    # Refractive Scattering
    #obs = obs.add_fractional_noise(0.02)
    obs = add_noisefloor_obs(obs, optype="quarter1", scale=1.0)
    # Diffractive Scattering
    sm = so.ScatteringModel()
    obs = sm.Deblur_obs(obs)

obs = obs.avg_coherent(60.0)
obs = obs.add_fractional_noise(0.01)

if args.pol=='I':
    clphs = pd.DataFrame(obs.c_phases(count='max', vtype='vis'))
elif args.pol=='Q':
    clphs = pd.DataFrame(obs.c_phases(count='max', vtype='qvis'))
elif args.pol=='U':
    clphs = pd.DataFrame(obs.c_phases(count='max', vtype='uvis'))
elif args.pol=='V':
    clphs = pd.DataFrame(obs.c_phases(count='max', vtype='vvis'))
else:
    print('Parse vaild pol string value: I, Q, U, V')

obs.add_scans()
times = []
for t in obs.scans:
    times.append(t[0])
obslist = obs.split_obs()
######################################################################
outpath = args.outpath
pol = args.pol

paths={}

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
   
    
polpaths={}
for p in paths.keys():
    mv=eh.movie.load_hdf5(paths[p])
    im=mv.get_image(times[0])
    
    if pol=='I':
       if len(im.ivec)>0:
            polpaths[p]=paths[p]
    elif pol=='Q':
        if len(im.qvec)>0:
            polpaths[p]=paths[p]
    elif pol=='U':
        if len(im.uvec)>0:
            polpaths[p]=paths[p]
    elif pol=='V':
        if len(im.vvec)>0:
            polpaths[p]=paths[p]
    else:
        print('Parse a vaild pol value')
    
colors = {   
            'kine'     : 'xkcd:azure',
            'ehtim'    : 'forestgreen',
            'doghit'   : 'darkviolet',
            'ngmem'    : 'crimson',
            'resolve'  : 'tab:orange'
        }

labels = {  
            'kine'     : 'kine',
            'ehtim'    : 'ehtim',
            'doghit'   : 'DoG-HiT',
            'ngmem'    : 'ngMEM',
            'resolve'  : 'resolve'
        }


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

tri_list = [('AZ', 'LM', 'SM'), ('AA', 'AZ', 'SM'), ('AA', 'LM', 'SM')]


######################################################################
mov_clphs = dict()

for p in polpaths.keys():
    mv = eh.movie.load_hdf5(polpaths[p])
    
    clphs_mod_time = dict()
    clphs_mod_win = dict()

    for tri in tri_list:
        clphs_time, clphs_window = [], []
        for ii in range(len(times)):
            tstamp = times[ii]
            im = mv.get_image(times[ii])
            im.rf = obslist_t[ii].rf
            if im.xdim%2 == 1:
                im = im.regrid_image(targetfov=im.fovx(), npix=im.xdim-1)
            obs_mod = im.observe_same(obslist_t[ii], add_th_noise=False, ttype='fast')

            # closure phase
            if args.pol=='I':
                clphs_mod = pd.DataFrame(obs_mod.c_phases(count='max', vtype='vis'))
            elif args.pol=='Q':
                clphs_mod = pd.DataFrame(obs_mod.c_phases(count='max', vtype='qvis'))
            elif args.pol=='U':
                clphs_mod = pd.DataFrame(obs_mod.c_phases(count='max', vtype='uvis'))
            elif args.pol=='V':
                clphs_mod = pd.DataFrame(obs_mod.c_phases(count='max', vtype='vvis'))
            else:
                print('Parse vaild pol string value: I, Q, U, V')
    

            # select triangle
            subtab  = select_triangle(clphs_mod, tri[0], tri[1], tri[2])
            try:
                idx = np.where(np.round(subtab['time'].values,3)  == np.round(tstamp,3))[0][0]                
                clphs_time.append(subtab['time'][idx])
                clphs_window.append(subtab['cphase'][idx])
            except:
                pass

        clphs_mod_time[tri] = clphs_time
        clphs_mod_win[tri] = clphs_window
        
    mov_clphs[p] = [clphs_mod_time, clphs_mod_win]
######################################################################

ctab = copy(clphs)
fix_xax = True

numplt = 3
xmin = min(ctab['time'].values)
xmax = max(ctab['time'].values)

fig = plt.figure(figsize=(21,4))
fig.subplots_adjust(wspace=0.3)

axs = []
for i in range(1,numplt+1):
    axs.append(fig.add_subplot(1,3,i))

for i in tqdm(range(numplt)):
    # closure phase
    subtab  = select_triangle(ctab, tri_list[i][0], tri_list[i][1], tri_list[i][2])
    axs[i].errorbar(subtab['time'], subtab['cphase'], yerr=subtab['sigmacp'],
                    c='black', mec='black', marker='o', ls="None", ms=5, alpha=0.5)
    
    # Model
    for pipe in polpaths.keys():
        mv = eh.movie.load_hdf5(polpaths[p])
        clphs_mod_time, clphs_mod_win = mov_clphs[pipe]

        # plot
        axs[i].errorbar(clphs_mod_time[tri_list[i]], clphs_mod_win[tri_list[i]], 
                        c=colors[pipe], marker='o', ms=2.5, ls="none", label=labels[pipe], alpha=0.5)
    

    axs[i].set_title("%s-%s-%s" %(tri_list[i][0], tri_list[i][1], tri_list[i][2]), fontsize=18)
    #if fix_yax:
    axs[i].set_ylim(-200, 200)
    #axs[i].grid()
    if i == 0:
        axs[i].legend(ncols=6, loc='best',  bbox_to_anchor=(3, 1.3), markerscale=5.0)
    axs[i].set_xlabel('Time (UTC)')
    axs[i].set_ylabel('cphase')

    if fix_xax:
        axs[i].set_xlim(xmin-0.5, xmax+0.5)

axs[0].text(10.5, 260, f'Stokes: {pol}', color='black', fontsize=18)

fig.subplots_adjust(top=0.93)
plt.savefig(args.outpath+'.png', bbox_inches='tight', dpi=300)