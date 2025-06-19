from __future__ import print_function, division
import numpy as np
import sys
import os
import itertools
import h5py
from mpi4py import MPI

import time

from halotools.sim_manager import CachedHaloCatalog

from halotools.empirical_models import HodModelFactory
from halotools.empirical_models.ia_models.ia_model_components import CentralAlignment, RadialSatelliteAlignment
from halotools.empirical_models.ia_models.ia_strength_models import RadialSatelliteAlignmentStrength
from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens, Zheng07Sats, SubhaloPhaseSpace
from halotools.mock_observables import tpcf
from halotools.mock_observables.ia_correlations import ee_3d, ed_3d

import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")

############################################################################################################################
##### FUNCTIONS ############################################################################################################
############################################################################################################################
# Eliminate halos with 0 for halo_axisA_x(,y,z)
def mask_bad_halocat(halocat):
    bad_mask = (halocat.halo_table["halo_axisA_x"] == 0) & (halocat.halo_table["halo_axisA_y"] == 0) & (halocat.halo_table["halo_axisA_z"] == 0)
    bad_mask = bad_mask ^ np.ones(len(bad_mask), dtype=bool)
    halocat._halo_table = halocat.halo_table[ bad_mask ]

def build_model_instance(cen_strength, sat_params, sat_bins, halocat, constant=True, seed=None):

    sat_alignment_strength = 1

    if constant:
        sat_alignment_strength = sat_params
    else:
        sat_a, sat_gamma = sat_params

    cens_occ_model = Zheng07Cens()
    cens_prof_model = TrivialPhaseSpace()
    cens_orientation = CentralAlignment(central_alignment_strength=cen_strength)

    sats_occ_model = Zheng07Sats()
    prof_args = ("satellites", sat_bins)
    sats_prof_model = SubhaloPhaseSpace(*prof_args)

    sats_orientation = RadialSatelliteAlignment(satellite_alignment_strength=sat_alignment_strength, halocat=halocat)
    if not constant:
        sats_strength = RadialSatelliteAlignmentStrength(satellite_alignment_a=sat_a, satellite_alignment_gamma=sat_gamma)
        Lbox = halocat.Lbox
        sats_strength.inherit_halocat_properties(Lbox=Lbox)
    
    if constant:
        model_instance = HodModelFactory(centrals_occupation = cens_occ_model,
                                        centrals_profile = cens_prof_model,
                                        satellites_occupation = sats_occ_model,
                                        satellites_profile = sats_prof_model,
                                        #satellites_radial_alignment_strength = sats_strength,
                                        centrals_orientation = cens_orientation,
                                        satellites_orientation = sats_orientation,
                                        model_feature_calling_sequence = (
                                        'centrals_occupation',
                                        'centrals_profile',
                                        'satellites_occupation',
                                        'satellites_profile',
                                        #'satellites_radial_alignment_strength',
                                        'centrals_orientation',
                                        'satellites_orientation')
                                        )
    else:
        model_instance = HodModelFactory(centrals_occupation = cens_occ_model,
                                        centrals_profile = cens_prof_model,
                                        satellites_occupation = sats_occ_model,
                                        satellites_profile = sats_prof_model,
                                        satellites_radial_alignment_strength = sats_strength,
                                        centrals_orientation = cens_orientation,
                                        satellites_orientation = sats_orientation,
                                        model_feature_calling_sequence = (
                                        'centrals_occupation',
                                        'centrals_profile',
                                        'satellites_occupation',
                                        'satellites_profile',
                                        'satellites_radial_alignment_strength',
                                        'centrals_orientation',
                                        'satellites_orientation')
                                        )

    model_instance.populate_mock(halocat,seed=seed)
    
    return model_instance

def build_specific_model_instance(central_alignment_strength, satellite_alignment_strength, logMmin, sigma_logM, logM0, logM1, alpha,
                                  sat_bins, halocat, seed=None):
    cens_occ_model = Zheng07Cens()
    cens_occ_model.param_dict["logMmin"] = logMmin
    cens_occ_model.param_dict["sigma_logM"] = sigma_logM
    cens_prof_model = TrivialPhaseSpace()
    cens_orientation = CentralAlignment(central_alignment_strength=central_alignment_strength)

    sats_occ_model = Zheng07Sats()
    sats_occ_model.param_dict["logM0"] = logM0
    sats_occ_model.param_dict["logM1"] = logM1
    sats_occ_model.param_dict["alpha"] = alpha
    prof_args = ("satellites", sat_bins)
    sats_prof_model = SubhaloPhaseSpace(*prof_args)
    sats_orientation = RadialSatelliteAlignment(satellite_alignment_strength=satellite_alignment_strength, halocat=halocat)

    model = HodModelFactory(centrals_occupation = cens_occ_model,
                                        centrals_profile = cens_prof_model,
                                        satellites_occupation = sats_occ_model,
                                        satellites_profile = sats_prof_model,
                                        centrals_orientation = cens_orientation,
                                        satellites_orientation = sats_orientation,
                                        model_feature_calling_sequence = (
                                        'centrals_occupation',
                                        'centrals_profile',
                                        'satellites_occupation',
                                        'satellites_profile',
                                        'centrals_orientation',
                                        'satellites_orientation')
                                        )
    model.populate_mock(halocat, seed=seed)
    return model

def correlate(row):
    func, args, kwargs = row
    return func(*args, **kwargs)
    
def generate_data(model_dict, halocat, input_dict, rbins, f_name, input_num, runs=10, max_attempts=5,
             output_dir="subsets", processes=5, parallel_method="iteration",
             store_columns=False, store_correlations=True, column_labels=None):

    results = iter_all(model_dict, halocat, input_dict, rbins, runs=runs, max_attempts=max_attempts,
                        processes=processes, parallel_method=parallel_method,
                        store_columns=store_columns, store_correlations=store_correlations, column_labels=column_labels)
    
    # Save the results
    # The file should already exist, made by the task calling this
    with h5py.File(os.path.join(output_dir, f_name), "a") as f:
        # Create a group for the input number
        if f"input_{input_num}" not in f:
            f.create_group(f"input_{input_num}")
        
        # Add the input parameters as attributes
        grp = f[f"input_{input_num}"]
        for key in input_dict:
            grp.attrs[key] = input_dict[key]
        # Add the results. Each iteration will be a new group, and each may contain the table subset and/or correlations
        for i in range(len(results)):
            # Create a group for the iteration
            if f"iteration_{i}" not in grp:
                grp.create_group(f"iteration_{i}")
            iter_grp = grp[f"iteration_{i}"]
            table, corrs = results[i]
            # Store the table subset if requested
            if store_columns:
                iter_grp.create_dataset("table_subset", data=table)
            # Store the correlations if requested
            if store_correlations:
                iter_grp.create_dataset("correlations", data=corrs)

def one_pass(model_dict, halocat, input_dict, rbins,
             processes=5, parallel_method="iteration",
             store_columns=False, store_correlations=True, column_labels=None):
    
    sat_bins = model_dict['sat_bins']
    seed = model_dict['seed']

    # Build model instance
    param_dict = dict(input_dict)
    param_dict["sat_bins"] = sat_bins
    param_dict["halocat"] = halocat
    param_dict["seed"] = seed
    model = build_specific_model_instance(**param_dict)

    table = []
    corrs = []

    if store_columns:
        # Assume not None by this point in the code
        table = model.mock.galaxy_table[column_labels]
    if store_correlations:
        # Calculate correlations
        corrs = corr_all(model, rbins, halocat, parallel=(parallel_method=="correlation"), processes=processes)

    return table, corrs

def iter_all(model_dict, halocat, input_dict, rbins, runs=10, max_attempts=5, 
             processes=5, parallel_method="iteration",
             store_columns=False, store_correlations=True, column_labels=None):
    
    results = [ () for _ in range(runs) ]  # Initialize results as a list of empty tuples

    if parallel_method == "correlation":
        for i in range(runs):
            repeat = True
            attempt = 0
            while repeat and attempt < max_attempts:
                table, corrs = one_pass(model_dict, halocat, input_dict, rbins, processes=processes, parallel_method=parallel_method,
                                        store_columns=store_columns, store_correlations=store_correlations, column_labels=column_labels)
                attempt += 1

                # Check for nans
                repeat = ( any( np.isnan(corrs[0]) ) or any( np.isnan(corrs[1]) ) or any( np.isnan(corrs[2]) ) )
            results.append((table, corrs))

    elif parallel_method == "iteration":
        if processes is None:
            processes = min(mp.cpu_count(), runs)           # Use all available cores, but not more than the number of runs

        # Build empty array to hold the results
        # Full array is mxcxb
        # m = number of different runs
        # c = number of different correlation functions (3)
        # b = number of different bins (rbins-1)
        repeat = np.ones(runs, dtype=bool)                # Array to keep track of which runs need to be repeated
        attempt = 0

        # Here is the parallelized loop
        # Perform this up to max_Attempts times
        # Replacing the results array with the new results where repeat is true
        while any(repeat) and attempt < max_attempts:

            rows = [ (model_dict, halocat, input_dict, rbins, processes, parallel_method, store_columns, store_correlations, column_labels) for _ in range(sum(repeat)) ]

            with mp.Pool(processes=processes) as pool:
                temp_results = pool.starmap(one_pass, rows)

            # Because results is a list of tuples where the first element is non-uniform shape, we have to store differently
            # Since we're only calculating a smaller subset, the indices won't match. Manually increment the temp_results index
            idx = 0
            for i in range(len(repeat)):
                if repeat[i]:
                    # Only replace the ones that have been repeated this round
                    results[i] = temp_results[idx]
                    idx += 1

            # Check for nans
            # If any of the results are nan, set repeat to true for that index
            corr_results = [ res[1] for res in results ]  # Extract the correlations from the results
            repeat = np.isnan(corr_results).any(axis=(1,2))        # Check for nans in the results array

            # Update attempt counter
            attempt += 1

    return results

def corr_all(model, rbins, halocat, parallel=False, processes=3):
    coords = np.array( [ model.mock.galaxy_table["x"], model.mock.galaxy_table["y"], model.mock.galaxy_table["z"] ] ).T
    orientations = np.array( [ model.mock.galaxy_table["galaxy_axisA_x"], 
                               model.mock.galaxy_table["galaxy_axisA_y"], 
                               model.mock.galaxy_table["galaxy_axisA_z"] ] ).T
    
    # These can be added to and expanded upon for more correlation types
    func_params = [
            ( tpcf, (coords, rbins, coords), {"period":halocat.Lbox} ),
            ( ed_3d, (coords, orientations, coords, rbins), {"period":halocat.Lbox} ),
            ( ee_3d, (coords, orientations, coords, orientations, rbins), {"period":halocat.Lbox} ),
    ]

    if parallel:
        with mp.Pool(processes=processes) as pool:
            results = pool.map(correlate, func_params)
    else:
        results = []
        for func, args, kwargs in func_params:
            results.append(func(*args, **kwargs))

    return np.array(results)