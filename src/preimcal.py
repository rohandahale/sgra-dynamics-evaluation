#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ehtim as eh
import numpy as np
import pandas as pd
import pickle
import scipy.interpolate as interp
import ehtim.imaging.dynamical_imaging as di
import ehtim.scattering.stochastic_optics as so

def add_noisefloor_obs(obs, optype="quarter1", scale=1.0):
    """ Function to add noisefloor to Obsdata.
    Requirements: numpy, scipy, pandas, ehtim;
    'obs' should be an ehtim.obsdata.Obsdata object
    Options for noisefloor include:
    'dime': add a constant value of (10 mJy * scale) in quadrature to all (u,v) data;
    'quarter1': (u,v)-dependent noisefloor values based on the RMS renormalized
                refractive noise values of a circular Gaussian model;
    'quarter2': (u,v)-dependent noisefloor values based on the averaged RMS renormalized
                refractive noise values of several synthetic models (ring, crescent, GRMHD, Gaussian).
    """
    epochs = {57849: '3598', 57850: '3599', 57854: '3601'}
    epoch = epochs[obs.mjd]
    if optype == "quarter1":
        # pickle file for the noise floor template
        infile = "./obs_scatt_std_%s.pickle" % (epoch)
        obs_new = add_noise_floor_obs_quarter(
            obs, infile=infile, scale=scale / 2.27)
    elif optype == "quarter2":
        # pickle file for the noise floor template
        infile = "./obs_scatt_std_%s.pickle" % (epoch)
        obs_new = add_noise_floor_obs_quarter(
            obs, infile=infile, scale=scale / 2.27)
    elif optype == "dime":   # constant noise floor of 10 mJy * scale
        obs_new = obs.copy()
        noise_floor = 1e-2 * scale / 2.27
        ref_noise = noise_floor * np.ones(obs_new.data['sigma'].shape)
        # add refractive noise in quadrature
        obs_new.data['sigma'] = np.sqrt(
            obs_new.data['sigma']**2 + ref_noise**2)
        obs_new.data['qsigma'] = np.sqrt(
            obs_new.data['qsigma']**2 + ref_noise**2)
        obs_new.data['usigma'] = np.sqrt(
            obs_new.data['usigma']**2 + ref_noise**2)
        obs_new.data['vsigma'] = np.sqrt(
            obs_new.data['vsigma']**2 + ref_noise**2)
    else:
        print("optype %s is not supported!" % (optype))
        obs_new = obs.copy()

    return obs_new


def add_noise_floor_obs_quarter(obs, infile, scale=1.0):
    """
    Add u,v dependent noisefloor to obs data by loading from a template
    and interpolate the noise values to match the shape of the input Obsdata.
    the templated file is in ehtim.obsdata.Obsdata format with
    the renormalized refractive noise values stored in the "vis" column

    Args:
        obs: the input obsdata
        infile: Pickle file for the model noise-floor table
        scale (float, optional):
            The scaling factor of the noise floor.
            Defaults to 1.0.

    Returns:
        obsdata: scattering-budget added obsfile.
    """
    # read obs.data and convert to a pandas dataframe
    df = pd.DataFrame(obs.data)

    # load the noise floor template
    pfile = open(infile, "rb")
    temp = pickle.load(pfile)
    nf = pd.DataFrame(temp.data)

    # interpolate the noise floor values and add to sigma for each baseline
    blgroup = df.groupby(['t1', 't2'])    # group by baseline

    df_new = pd.DataFrame()
    for key in list(blgroup.groups.keys()):
        # single baseline data
        dt = blgroup.get_group(key)
        try:
            # get the noise floor and time stamp from the template
            nfl_org = nf.groupby(['t1', 't2']).get_group(key)["vis"]
            t_org = nf.groupby(['t1', 't2']).get_group(key)["time"]
        except KeyError:
            key2 = (key[1], key[0])
            # get the noise floor and time stamp from the template
            nfl_org = nf.groupby(['t1', 't2']).get_group(key2)["vis"]
            t_org = nf.groupby(['t1', 't2']).get_group(key2)["time"]

        # interpolate
        f = interp.interp1d(t_org, nfl_org, fill_value="extrapolate")
        t_new = dt['time']
        nfl_new = f(t_new) * scale

        # add noise floor
        dt.loc[:, "sigma"] = np.sqrt(
            dt["sigma"]**2 + np.abs(nfl_new)**2) * np.sign(dt["sigma"])
        dt.loc[:, "qsigma"] = np.sqrt(
            dt["qsigma"]**2 + np.abs(nfl_new)**2) * np.sign(dt["qsigma"])
        dt.loc[:, "usigma"] = np.sqrt(
            dt["usigma"]**2 + np.abs(nfl_new)**2) * np.sign(dt["usigma"])
        dt.loc[:, "vsigma"] = np.sqrt(
            dt["vsigma"]**2 + np.abs(nfl_new)**2) * np.sign(dt["vsigma"])

        # append the new data of the baseline to the new dataframe
        df_new = df_new.append(dt)

    df_new = df_new.sort_index()    # restore the original order of the data

    # write the new sigma to data
    obs_new = obs.copy()
    obs_new.data["sigma"] = df_new["sigma"]
    obs_new.data["qsigma"] = df_new["qsigma"]
    obs_new.data["usigma"] = df_new["usigma"]
    obs_new.data["vsigma"] = df_new["vsigma"]

    return obs_new