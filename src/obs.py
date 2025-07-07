import os
import contextlib
import warnings
warnings.filterwarnings('ignore')

with contextlib.redirect_stdout(open(os.devnull, 'w')):
    import ehtim

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Set

class Inputs:
    def __init__(self, 
                 obs_true: ehtim.obsdata.Obsdata,
                 obs_recons: Dict[str, ehtim.obsdata.Obsdata],
                 mv_true: Optional[ehtim.movie.Movie] = None,
                 mv_recons: Optional[Dict[str, Union[ehtim.movie.Movie, List[ehtim.movie.Movie]]]] = None):
        """All inputs
        
        Args:
            obs_true: True/reference observation
            obs_recons: Dictionary mapping imaging method names to ehtim.obsdata.Obsdata objects
            mv_true: True/reference ehtim.movie.Movie object (optional)
            mv_recons: Dictionary mapping imaging method names to ehtim.movie.Movie object or a list of ehtim.movie.Movie objects (optional)
        """        
        self.obs_true = obs_true
        self.obs_recons = obs_recons
        self.mv_true = mv_true
        self.mv_recons = self._standardize_dict(mv_recons) if mv_recons is not None else None
        
        # Store method names
        self.method_names = set(obs_recons.keys())
        if mv_recons is not None:
            if set(mv_recons.keys()) != self.method_names:
                raise ValueError("Method names in obs_recons and mv_recons must match")
        
        self._validate_inputs()
        self.obs_times = self._get_observation_times()
        self.movie_times = self._get_movie_times() if mv_true is not None else None

    def _standardize_dict(self, data_dict: Dict) -> Dict:
        """Convert dictionary values to lists if they're single objects"""
        if data_dict is None:
            return None
        return {
            method: [data] if isinstance(data, (ehtim.obsdata.Obsdata, ehtim.movie.Movie)) else data
            for method, data in data_dict.items()
        }

    def _validate_inputs(self) -> None:
        """Validate input data formats and consistency"""
        if not isinstance(self.obs_true, ehtim.obsdata.Obsdata):
            raise TypeError("obs_true must be an ehtim Obsdata object")
        
        if not isinstance(self.obs_recons, dict):
            raise TypeError("obs_recons must be a dictionary")
            
        for method, obs in self.obs_recons.items():
            if not isinstance(obs, ehtim.obsdata.Obsdata):
                raise TypeError(f"Value for method {method} must be an Obsdata object")
                    
        if self.mv_true is not None:
            if not isinstance(self.mv_true, ehtim.movie.Movie):
                raise TypeError("mv_true must be an ehtim Movie object")
                
            if self.mv_recons is not None:
                if not isinstance(self.mv_recons, dict):
                    raise TypeError("mv_recons must be a dictionary")
                    
                for method, mov_list in self.mv_recons.items():
                    if not isinstance(mov_list, list):
                        raise TypeError(f"Value for method {method} must be a list")
                    for mov in mov_list:
                        if not isinstance(mov, ehtim.movie.Movie):
                            raise TypeError(f"All elements for method {method} must be Movie objects")

    def _get_observation_times(self) -> Dict[str, np.ndarray]:
        """Extract observation times for true and reconstructed data"""
        times = {
            'true': np.unique(self.obs_true.data['time'])
        }
        
        for method, obs in self.obs_recons.items():
            method_times = []
            method_times.extend(np.unique(obs.data['time']))
            times[method] = np.unique(method_times)
            
        return times
    
    def _get_movie_times(self) -> Dict[str, np.ndarray]:
        """Extract movie times for true and reconstructed data"""
        if self.mv_true is None:
            return None
            
        times = {
            'true': self.mv_true.times
        }
        
        for method, mov_list in self.mv_recons.items():
            method_times = []
            for mov in mov_list:
                method_times.extend(mov.times)
            times[method] = np.unique(method_times)
            
        return times
    
    def get_method_data(self, method: str) -> tuple:
        """Get all data for a specific reconstruction method
        
        Args:
            method: Name of the reconstruction method
            
        Returns:
            Tuple containing:
            - Obsdata object for the method
            - List of Movie objects for the method (if available)
        """
        if method not in self.method_names:
            raise ValueError(f"Unknown method: {method}")
            
        method_obs = self.obs_recons[method]
        method_movies = self.mv_recons[method] if self.mv_recons else None
        
        return method_obs, method_movies


class ChiSquare:
    """Compute chi-square statistics of reconstructions"""
    
    VALID_DTYPES = ['vis', 'amp', 'bs', 'cphase', 'logcamp']
    VALID_POLS = ['I', 'Q', 'U', 'V']
    
    def __init__(self, inputs: Inputs):
        """Initialize with Inputs object"""
        self.inputs = inputs

    def _crop_obs_times(self, obs: ehtim.obsdata.Obsdata, movie: ehtim.movie.Movie) -> ehtim.obsdata.Obsdata:
        """Align observation times with movie times"""
        movie_start = min(movie.times)
        movie_end = max(movie.times)
        return obs.flag_UT_range(UT_start_hour=movie_start, UT_stop_hour=movie_end, output='flagged')

    def compute_chisq(self, 
                     method: str,
                     dtype: str = 'vis',
                     pol: str = 'I',
                     systematic_noise: float = 0.0,
                     **kwargs) -> Dict[str, float]:
        """Compute chi-square for specified method and parameters"""
        if dtype not in self.VALID_DTYPES:
            raise ValueError(f"Invalid dtype. Must be one of {self.VALID_DTYPES}")
        if pol not in self.VALID_POLS:
            raise ValueError(f"Invalid pol. Must be one of {self.VALID_POLS}")
        if method != 'true' and method not in self.inputs.method_names:
            raise ValueError(f"Unknown method: {method}")

        if method == 'true':
            if self.inputs.mv_true is None:
                raise ValueError("No true movie available")
            movies = [self.inputs.mv_true]
            obs = self.inputs.obs_true
        else:
            obs = self.inputs.obs_recons[method]
            movies = self.inputs.mv_recons[method]
            if not isinstance(movies, list):
                movies = [movies]

        extra_args = {}
        if dtype == 'cphase':
            extra_args['cp_uv_min'] = kwargs.get('cp_uv_min', 0)
        elif dtype == 'logcamp':
            extra_args['cp_uv_min'] = kwargs.get('cp_uv_min', 0)
            extra_args['snrcut'] = kwargs.get('snrcut', 1.0)

        results = []
        for mov in movies:
            # Align observation times with movie times
            cropped_obs = self._crop_obs_times(obs, mov)
            chi2 = cropped_obs.chisq(
                mov,
                dtype=dtype,
                pol=pol,
                ttype='direct',
                systematic_noise=systematic_noise,
                **extra_args
            )
            results.append(chi2)

        return {dtype: np.mean(results)}

    def compute_all_dtypes(self, 
                        method: str,
                        **kwargs) -> Dict[str, Dict[str, float]]:
        """Compute chi-square for all data types and polarizations for a method"""
        results = {}
        for dtype in self.VALID_DTYPES:
            dtype_results = {}
            for pol in self.VALID_POLS:
                try:
                    chi2 = self.compute_chisq(method, dtype=dtype, pol=pol, **kwargs)
                    dtype_results[pol] = chi2[dtype]
                except Exception as e:
                    print(f"Warning: Failed to compute {dtype}/{pol} chi-square: {str(e)}")
            if dtype_results:
                results[dtype] = dtype_results
        return results

    def compute_all_methods(self, 
                        dtypes: Optional[List[str]] = None,
                        include_true: bool = True,
                        **kwargs) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute chi-square for all methods, data types and polarizations"""
        results = {}
        dtypes = dtypes or self.VALID_DTYPES
        
        methods = list(self.inputs.method_names)
        if include_true and self.inputs.mv_true is not None:
            methods.append('true')
            
        for method in methods:
            method_results = {}
            for dtype in dtypes:
                dtype_results = {}
                for pol in self.VALID_POLS:
                    try:
                        chi2 = self.compute_chisq(method, dtype=dtype, pol=pol, **kwargs)
                        dtype_results[pol] = chi2[dtype]
                    except Exception as e:
                        print(f"Warning: Failed to compute {dtype}/{pol} chi-square for {method}: {str(e)}")
                if dtype_results:
                    method_results[dtype] = dtype_results
            results[method] = method_results
            
        return results

basedir='/mnt/disks/shared/eht/sgra_dynamics_april11/DAR/submissions/'   
with contextlib.redirect_stdout(open(os.devnull, 'w')):    
    mv_true = ehtim.movie.load_hdf5(f'{basedir}/mring+hsCCW_LO_onsky_truth.hdf5')
    obs_true = ehtim.obsdata.load_uvfits(f'{basedir}/mring+hsCCW_LO_onsky.uvfits')
    kine = ehtim.movie.load_hdf5(f'{basedir}/mring+hsCCW_LO_onsky_kine.hdf5')
    resolve = ehtim.movie.load_hdf5(f'{basedir}/mring+hsCCW_LO+HI_onsky_resolve_mean.hdf5')

obs_recons = {'kine': obs_true, 'resolve': obs_true}
mv_recons = {'kine': kine, 'resolve': resolve}

mringhsccw = Inputs(obs_true=obs_true, obs_recons=obs_recons, mv_true=mv_true, mv_recons=mv_recons)

#method_obs, method_movies = mringhsccw.get_method_data('kine')
#print([len(mringhsccw.obs_times[method]) for method in mringhsccw.obs_times.keys()])
#print([len(mringhsccw.movie_times[method]) for method in mringhsccw.movie_times.keys()])
#print(mringhsccw.method_names)
#print(mringhsccw.get_method_data('kine'))


chisq_stats = ChiSquare(mringhsccw)
chisq_results = chisq_stats.compute_all_methods()
#chisq_results = chisq_stats.compute_all_dtypes('kine')
print(chisq_results)