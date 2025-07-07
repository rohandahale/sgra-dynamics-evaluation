######################################################################
# Author: Rohan Dahale, Date: 09 November 2024
######################################################################

import ehtim as eh
import ehtim.scattering.stochastic_optics as so
from preimcal import *
import numpy as np
import pdb
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
    p.add_argument('--kinemv', type=str, default='none', help='path of kine .hdf5')
    p.add_argument('--starmv', type=str, default='none', help='path of starwarps .hdf5')
    p.add_argument('--ehtmv',  type=str, default='none', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',  type=str, default='none', help='path of doghit .hdf5')
    p.add_argument('--ngmv',   type=str, default='none', help='path of ngmem .hdf5')
    p.add_argument('--resmv',  type=str, default='none',help='path of resolve .hdf5')
    p.add_argument('--modelingmv',  type=str, default='none', help='path of modeling .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./gif.gif', 
                   help='name of output file with path')
    p.add_argument('--scat', type=str, default='none', help='onsky, deblur, dsct, none')

    return p

# List of parsed arguments
args = create_parser().parse_args()

outpath = args.outpath

paths={}

if args.truthmv!='none':
    paths['truth']=args.truthmv
if args.kinemv!='none':
    paths['kine']=args.kinemv
if args.resmv!='none':
    paths['resolve']=args.resmv
if args.starmv!='none':
    paths['starwarps']=args.starmv
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
    
######################################################################
# Set parameters
npix   = 200
fov    = 200 * eh.RADPERUAS
blur   = 0 * eh.RADPERUAS
######################################################################

for p in paths.keys():
    if not os.path.exists(outpath+os.path.basename(paths[p])[:-5]+'.fits'):
        mov = eh.movie.load_hdf5(paths[p])
        imlist = []
        imlistIarr=[]
        imlistQarr=[]
        imlistUarr=[]
        imlistVarr=[]
        for t in times:
            im = mov.get_image(t)
            #if p=='truth':
            #    if args.scat!='onsky':
            #        im = im.blur_circ(fwhm_i=15*eh.RADPERUAS, fwhm_pol=15*eh.RADPERUAS).regrid_image(fov, npix)
            im = im.blur_circ(fwhm_i=blur, fwhm_pol=blur).regrid_image(fov, npix)
            imlist.append(im)
            imlistIarr.append(im.imarr(pol='I'))
            imlistQarr.append(im.imarr(pol='Q'))
            imlistUarr.append(im.imarr(pol='U'))
            imlistVarr.append(im.imarr(pol='V'))
            
        medianI = np.median(imlistIarr,axis=0)
        medianQ = np.median(imlistQarr,axis=0)
        medianU = np.median(imlistUarr,axis=0)
        medianV = np.median(imlistVarr,axis=0)
        
        #medianI = np.min(imlistIarr,axis=0)
        #medianQ = np.min(imlistQarr,axis=0)
        #medianU = np.min(imlistUarr,axis=0)
        #medianV = np.min(imlistVarr,axis=0)
        
        if len(imlist[0].ivec)!=0: 
            imlist[0].ivec = medianI.flatten()
        if len(imlist[0].qvec)!=0:
            imlist[0].qvec = medianQ.flatten()
        if len(imlist[0].uvec)!=0:
            imlist[0].uvec = medianU.flatten()
        if len(imlist[0].vvec)!=0:
            imlist[0].vvec = medianV.flatten()
        
        imlist[0].save_fits(outpath+os.path.basename(paths[p])[:-5]+'.fits')
        