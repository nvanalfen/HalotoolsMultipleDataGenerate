from __future__ import print_function, division
import numpy as np
import sys
import os
import itertools

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

def make_data_row(model, rbins, halocat):
    gal_table = model.mock.galaxy_table

    coords = np.array( [ gal_table["x"], gal_table["y"], gal_table["z"] ] ).T
    orientations = np.array( [ gal_table["galaxy_axisA_x"], gal_table["galaxy_axisA_y"], gal_table["galaxy_axisA_z"] ] ).T
    
    return (coords, orientations, rbins, halocat.Lbox)

def correlate(row):
    func, args, kwargs = row
    return func(*args, **kwargs)

def generate_correlations_parallel(model, rbins, halocat, processes=3):
    gal_table = model.mock.galaxy_table
    cen_cut = gal_table[ gal_table["gal_type"] == "centrals" ]
    sat_cut = gal_table[ gal_table["gal_type"] == "satellites" ]

    coords = np.array( [ gal_table["x"], gal_table["y"], gal_table["z"] ] ).T
    orientations = np.array( [ gal_table["galaxy_axisA_x"], gal_table["galaxy_axisA_y"], gal_table["galaxy_axisA_z"] ] ).T
    cen_coords = np.array( [ cen_cut["x"], cen_cut["y"], cen_cut["z"] ] ).T
    cen_orientations = np.array( [ cen_cut["galaxy_axisA_x"], cen_cut["galaxy_axisA_y"], cen_cut["galaxy_axisA_z"] ] ).T
    sat_coords = np.array( [ sat_cut["x"], sat_cut["y"], sat_cut["z"] ] ).T
    sat_orientations = np.array( [ sat_cut["galaxy_axisA_x"], sat_cut["galaxy_axisA_y"], sat_cut["galaxy_axisA_z"] ] ).T
    
    func_params = [
            ( tpcf, (coords, rbins, coords), {"period":halocat.Lbox} ),
            ( ed_3d, (coords, orientations, coords, rbins), {"period":halocat.Lbox} ),
            ( ee_3d, (coords, orientations, coords, orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (cen_coords, rbins, cen_coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (cen_coords, cen_orientations, cen_coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (cen_coords, cen_orientations, cen_coords, cen_orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (sat_coords, rbins, sat_coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (sat_coords, sat_orientations, sat_coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (sat_coords, sat_orientations, sat_coords, sat_orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (coords, rbins, cen_coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (coords, orientations, cen_coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (coords, orientations, cen_coords, cen_orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (cen_coords, rbins, coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (cen_coords, cen_orientations, coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (cen_coords, cen_orientations, coords, orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (coords, rbins, sat_coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (coords, orientations, sat_coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (coords, orientations, sat_coords, sat_orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (sat_coords, rbins, coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (sat_coords, sat_orientations, coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (sat_coords, sat_orientations, coords, orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (cen_coords, rbins, sat_coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (cen_coords, cen_orientations, sat_coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (cen_coords, cen_orientations, sat_coords, sat_orientations, rbins), {"period":halocat.Lbox} ),

            # ( tpcf, (sat_coords, rbins, cen_coords), {"period":halocat.Lbox} ),
            # ( ed_3d, (sat_coords, sat_orientations, cen_coords, rbins), {"period":halocat.Lbox} ),
            # ( ee_3d, (sat_coords, sat_orientations, cen_coords, cen_orientations, rbins), {"period":halocat.Lbox} )
            ]
    
    with mp.Pool(processes=processes) as pool:
        results = pool.map(correlate, func_params)
    
    return results

def iterate_wrapper(coords, orientations, rbins, Lbox):
    func_params = [
            ( tpcf, (coords, rbins, coords), {"period":Lbox} ),
            ( ed_3d, (coords, orientations, coords, rbins), {"period":Lbox} ),
            ( ee_3d, (coords, orientations, coords, orientations, rbins), {"period":Lbox} ),
    ]

    results = []
    for func, args, kwargs in func_params:
        results.append(func(*args, **kwargs))
    
    return np.array(results)

def generate_correlations_parallel_loop_series(model, rbins, halocat, runs, max_attempts, processes=3):
    """
    This function will do each loop in series, running all correlations in parallel each loop
    """
    # Build empty array to hold the results
    # Full array is Nxmxcxb
    # N = number of different inputs
    # m = number of different runs
    # c = number of different correlation functions (3)
    # b = number of different bins
    # In this function, we're only working with a single set of inputs, so we need mxcxb
    outputs = np.zeros( (runs, 3, len(rbins)-1) )

    # Repopulate and sample
    for i in range(runs):
        repeat = True
        attempt = 0

        while repeat and attempt < max_attempts:
            model.mock.populate()

            # Calculate correlations
            results = generate_correlations_parallel(model, rbins, halocat, processes=processes)
            attempt += 1

            # Check for nans
            repeat = ( any( np.isnan(results[0]) ) or any( np.isnan(results[1]) ) or any( np.isnan(results[2]) ) )

        outputs[i] = results

    return outputs

def generate_correlations_series_loop_parallel(model, rbins, halocat, runs, max_attempts, processes=None):
    """
    This function runs each loop in parallel with one another, doing the correlations serially inside each loop.
    Because the model instance of HODModelFactory is not pickleable, we need to iterate and populate to build a list of inputs for the pool.
    """
    if processes is None:
        processes = min(mp.cpu_count(), runs)           # Use all available cores, but not more than the number of runs

    # Build empty array to hold the results
    # Full array is mxcxb
    # m = number of different runs
    # c = number of different correlation functions (3)
    # b = number of different bins (rbins-1)
    results = np.zeros( (runs, 3, len(rbins)-1) )
    repeat = np.ones(runs, dtype=bool)                # Array to keep track of which runs need to be repeated
    attempt = 0

    # Here is the parallelized loop
    # Perform this up to max_Attempts times
    # Replacing the results array with the new results where repeat is true
    while any(repeat) and attempt < max_attempts:
        # Build list of positions and orientations to pass in.
        # Because model is NOT pickleable, we need to iterate and populate to build a list of inputs for the pool
        rows = []
        for i in range( sum(repeat) ):
            # Equal to number of run on first attempt
            # Any subsequent runs only need to create as many data rows as there are repeat==true indices
            model.mock.populate()
            rows.append(make_data_row(model, rbins, halocat))

        with mp.Pool(processes=processes) as pool:
            temp_results = pool.starmap(iterate_wrapper, rows)

        results[repeat] = np.array(temp_results)

        # Check for nans
        # If any of the results are nan, set repeat to true for that index
        repeat = np.isnan(results).any(axis=(1,2))        # Check for nans in the results array

        # Update attempt counter
        attempt += 1
        
    return results

def calculate_all_iterations(model, rbins, halocat, runs, input_dict, max_attempts, processes=3, parallel_method="correlation"):
    
    assert parallel_method in ["correlation", "iteration"], "parallel_method must be either 'correlation' or 'iteration'"

    # Adjust model params
    for key in input_dict.keys():
        model.param_dict[key] = input_dict[key]

    # Full array is Nxmxcxb
    # N = number of different inputs
    # m = number of different runs
    # c = number of different correlation functions (3)
    # b = number of different bins
    # In this function, we're only working with a single set of inputs, so we need mxcxb
    output_shape = (runs, 3, len(rbins)-1)

    try:
        if parallel_method == "correlation":
            return generate_correlations_parallel_loop_series(model, rbins, halocat, runs, max_attempts, processes=processes)
        elif parallel_method == "iteration":
            return generate_correlations_series_loop_parallel(model, rbins, halocat, runs, max_attempts, processes=processes)
    except:
        print(f"Failed on {input_dict}", flush=True)
        return np.zeros(output_shape)              # Return all zeros in the case of a catastophic failure

def generate_training_data(model, rbins, job, max_jobs, halocat, inner_runs=10, save_every=5, 
                           param_loc="params.npz", output_dir="data", suffix="", max_attempts=5):
    """
    Generate training data using a particular model instance and halo cat, assuming one of several slurm jobs being run.

    Parameters
    ----------
    model : HodModelFactory
        The model instance to use for generating the data.
    rbins : array_like
        The radial bins to use for the correlation functions.
    job : int
        The job number for the slurm job.
    max_jobs : int
        The total number of slurm jobs being run.
    halocat : CachedHaloCatalog
        The halo catalog to use for generating the data.
    inner_runs : int, optional
        The number of times to run the model for each set of parameters. Default is 10.
    save_every : int, optional
        The number of iterations to run before saving the data. Default is 5.
    param_loc : str, optional
        The location of the input parameter file. Default is "params.npz".
    output_dir : str, optional
        The location to save the output data (along with a copy of the inputs used for this particular section). Default is "data".
    suffix : str, optional
        A suffix to add to the output file names. Default is "".
    max_attempts : int, optional
        The maximum number of attempts to run the model for each set of parameters. Default is 5. If the model fails to run after 
        this many attempts, proceed with caution.

    Returns
    -------
    keys : array_like
        The keys used for the input parameters.
    inputs : array_like
        The input parameters used for generating the data.
    outputs : array_like - shape (N, inner_runs, 3, len(rbins)-1)
        The output data generated by the model.
    """

    # Get the section of the full logMmin to run
    # span = int(np.ceil(Npts/max_jobs))
    # start = (job-1)*span
    # end = start+span

    # Get values
    param_arr = np.load(param_loc, allow_pickle=True)
    all_inputs = param_arr["values"]
    usable_inputs = all_inputs[job-1::max_jobs]                                 # Only access every max_jobs-th value

    # Make an empty array for the outputs (inputs kept as a list to keep track of progress)
    inputs = []
    outputs = np.zeros((len(usable_inputs), inner_runs, 3, len(rbins)-1))

    start_point = 0
    if os.path.exists( os.path.join( output_dir, f"full_run_{suffix}.npz" ) ):
        data = np.load( os.path.join( output_dir, f"full_run_{suffix}.npz" ), allow_pickle=True )
        inputs = data["inputs"].tolist()
        outputs = data["outputs"]
        start_point = len(inputs)
        print("Loaded existing data", flush=True)

    keys = np.array(['central_alignment_strength', 'satellite_alignment_strength',
                        'logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha'])

    for i in range(len(usable_inputs))[start_point:]:
        print(f"Starting {i}", flush=True)

        input_dict = {keys[j]: usable_inputs[i][j] for j in range(len(keys))}
        results = calculate_all_iterations(model, rbins, halocat, inner_runs, input_dict, max_attempts)
        inputs.append( usable_inputs[i] )
        outputs[i] = results

        if i % save_every == 0:
            np.savez( os.path.join( output_dir, f"full_run_{suffix}.npz" ), keys=keys, inputs=inputs, outputs=outputs )

    return keys, np.array(inputs), outputs

def main(job, max_jobs):
    ############################################################################################################################
    # MODEL PARAMETERS #########################################################################################################
    ############################################################################################################################
    inner_runs = 10
    constant = True
    catalog = "bolplanck"
    #catalog = "multidark"
    # Set rbins - Larger max distance means longer run time (from correlation calculations)
    rbins = np.logspace(-1,1.2,21)
    #rbins = np.logspace(-1,1.8,29)
    rbin_centers = (rbins[:-1]+rbins[1:])/2.0

    param_f_name = "test_params.npz"                # Location of the input parameter file
    output_dir = "results"                     # Location to save the output data

    ############################################################################################################################

    # Initial strength parameters
    # These don't matter, as they are only needed for the initial creation of the model
    central_alignment_strength = 1
    satellite_alignment_strength = 1

    # Satellite bins
    sat_bins = np.logspace(10.5, 15.2, 15)
    if catalog == "multidark":
        sat_bins = np.logspace(12.4, 15.5, 15)

    # Set up halocat
    halocat = CachedHaloCatalog(simname=catalog, halo_finder='rockstar', redshift=0, version_name='halotools_v0p4')
    mask_bad_halocat(halocat)

    start = time.time()

    suffix = ("constant" if constant else "distance_dependent") + "_" + catalog + ("_"+str(job) if not job is None else "")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build model instance
    model = build_model_instance(central_alignment_strength, satellite_alignment_strength, sat_bins, 
                                halocat, constant=constant, seed=None)

    # Generate Data
    keys, inputs, outputs = generate_training_data(model, rbins, job, max_jobs, halocat, 
                                            inner_runs=inner_runs, save_every=2, param_loc=param_f_name, output_dir=output_dir, suffix=suffix)

    # Save data, making sure to account for this script being run on multiple jobs
    np.savez( os.path.join( output_dir, f"full_run_{suffix}.npz" ), keys=keys, inputs=inputs, outputs=outputs )

    print("Time: ", time.time()-start,"\n", flush=True)

if __name__ == "__main__":
    ############################################################################################################################
    ##### SET UP VARIABLES #####################################################################################################
    ############################################################################################################################

    # Administrative variables
    assert len(sys.argv) == 3, "Must provide job number and max jobs"
    job = int(sys.argv[1])
    max_jobs = int(sys.argv[2])

    main(job, max_jobs)