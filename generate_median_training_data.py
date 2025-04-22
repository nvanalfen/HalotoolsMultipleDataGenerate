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

def correlate(row):
    func, args, kwargs = row
    return func(*args, **kwargs)

def generate_correlations_parallel(model, rbins, halocat):
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
    
    with mp.Pool() as pool:
        results = pool.map(correlate, func_params)
    
    return results

def generate_training_data(model, rbins, job, max_jobs, halocat, inner_runs=10, save_every=5, 
                           param_loc="params.npz", output_dir="data", suffix=""):
    inputs = []
    outputs = {}

    # Get the section of the full logMmin to run
    # span = int(np.ceil(Npts/max_jobs))
    # start = (job-1)*span
    # end = start+span

    # Get values
    param_arr = np.load(param_loc, allow_pickle=True)
    all_inputs = param_arr["values"]
    usable_inputs = all_inputs[job-1::max_jobs]                                 # Only access every max_jobs-th value

    start = time.time()

    start_point = 0
    if os.path.exists( os.path.join( output_dir, f"inputs_{suffix}.npy" ) ):
        inputs = np.load( os.path.join( output_dir, f"inputs_{suffix}.npy" ), allow_pickle=True ).tolist()
        outputs = np.load( os.path.join( output_dir, f"outputs_{suffix}.npy" ), allow_pickle=True ).item()
        start_point = len(inputs)
        print("Loaded existing data")

    keys = np.array(['central_alignment_strength', 'satellite_alignment_strength',
                        'logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha'])

    ind = len(inputs)
    max_attempts = 5        # Maximum number of times to build a model without nans before giving up and moving on

    # If the latest run didn't finish, start one run earlier
    fresh_run = True
    if len(inputs) != 0 and len(outputs[ tuple(inputs[-1]) ]) < inner_runs:
        fresh_run = False
        start_point -= 1

    for i in range(len(usable_inputs))[start_point:]:
        print(f"Starting {i}")

        # Add to the inputs list
        if fresh_run:
            # Only append if we are starting from scratch
            inputs.append( usable_inputs[i] )
        else:
            # If the inputs are already included, don't append again
            # Do set fresh_run to True so that we don't skip the next set of parameters
            fresh_run = True

        inner_adjust = 0

        # If this is a new set, add an empty list to the outputs
        if not tuple( usable_inputs[i] ) in outputs.keys():
            outputs[ tuple( usable_inputs[i] ) ] = []
        else:
            # Otherwise, see how many we've done already and use that to lessen the remaining number of inner runs
            inner_adjust = len(outputs[ tuple( usable_inputs[i] ) ])
        
        # If we've somehow already done as many as needed, skip this set of parameters
        if inner_adjust >= inner_runs:
            continue

        # Adjust model params
        for j in range(len(keys)):
            model.param_dict[keys[j]] = usable_inputs[i][j]             # Since values are stored in the same order as the keys, this works

        # Repopulate and sample
        for j in range(inner_runs-inner_adjust):
            try:
                repeat = True
                attempt = 0

                while repeat and attempt < max_attempts:
                    model.mock.populate()

                    # Calculate correlations
                    results = generate_correlations_parallel(model, rbins, halocat)
                    attempt += 1

                    # Check for nans
                    repeat = ( any( np.isnan(results[0]) ) or any( np.isnan(results[1]) ) or any( np.isnan(results[2]) ) )

                outputs[ tuple(usable_inputs[i]) ].append( results )

                if i % save_every == 0:
                    np.save( os.path.join( output_dir, f"inputs_{suffix}.npy" ), inputs)
                    np.save( os.path.join( output_dir, f"outputs_{suffix}.npy" ), outputs)
            except:
                print(f"Failed on {i}")

        print(f"{ind} - {time.time()-start}")
        ind += 1

    return keys, np.array(inputs), outputs

############################################################################################################################
##### SET UP VARIABLES #####################################################################################################
############################################################################################################################

# Administrative variables
assert len(sys.argv) == 3, "Must provide job number and max jobs"
job = int(sys.argv[1])
max_jobs = int(sys.argv[2])

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

param_f_name = "params.npz"                # Location of the input parameter file
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
                                        inner_runs=inner_runs, save_every=5, param_loc="../100X_Data/params.npz", output_dir=output_dir, suffix=suffix)

# Save data, making sure to account for this script being run on multiple jobs
np.save( os.path.join( output_dir, f"inputs_{suffix}.npy" ), inputs)
np.save( os.path.join( output_dir, f"outputs_{suffix}.npy" ), outputs)

# if job is None or job == 1:
#     summary_file = open( os.path.join( output_dir, f"summary_{suffix}.txt" ), "w" )
#     for key in parameters.keys():
#         summary_file.write( f"{key}: {parameters[key]}\n" )
#     summary_file.write( f"Time: {time.time()-start}\n" )
#     summary_file.close()

print("Time: ", time.time()-start,"\n")