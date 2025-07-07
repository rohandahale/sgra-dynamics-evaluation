######################################################################
# Author: Rohan Dahale, Date: 11 November 2024
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
import scipy
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

pathmovt  = args.truthmv
outpath = args.outpath

npix   = 200
fov    = 200 * eh.RADPERUAS

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
if args.modelingmv!='none':
    paths['modeling']=args.modelingmv
    
#################################################################################################

obs = eh.obsdata.load_uvfits(args.data)
obs, obs_t, obslist_t, splitObs, times, I, snr, w_norm = process_obs_weights(obs, args, paths)

##################################################################################################

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24,8))

ax[0].set_ylabel('Relative to Maximum')
ax[1].set_ylabel('Percentage')

ax[0].set_xlabel('Time (UTC)')
ax[1].set_xlabel('Time (UTC)')

ax[0].plot(times, I, label='Istropy Metric (I)', marker ='o', ms=5, ls='-')
ax[0].plot(times, snr['I'], label='SNR', marker ='o', ms=5, ls='-')
ax[0].plot(times, I*snr['I'], label='I x SNR', marker ='o', ms=5, ls='-')
ax[0].set_title('Data Quality: I, SNR', fontsize=22)

ax[1].plot(times, w_norm['I']*100, label='Weights', marker ='o', ms=5, ls='-')
ax[1].hlines(100/np.size(w_norm['I']), times[0], times[-1], color='black', linestyle='--')
ax[1].set_title('Weights: Istropy Metric x SNR', fontsize=22)


ax[0].legend(loc='best')
plt.savefig(args.outpath+'.png', bbox_inches='tight', dpi=300)

df = pd.DataFrame({"time": times, 
                   "weighs_I" :  w_norm['I'], 
                   "weights_Q" :  w_norm['Q'], 
                   "weights_U" :  w_norm['U'],
                   "isotropy_metric" : I,
                   "snr_I" : snr['I'],
                   "snr_Q" : snr['Q'],
                   "snr_U" : snr['U']})
df.to_csv(args.outpath+'.csv', index=False)