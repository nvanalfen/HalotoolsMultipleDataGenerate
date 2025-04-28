import numpy as np
import multiprocessing as mp
from halotools.sim_manager import CachedHaloCatalog

from halotools.empirical_models import HodModelFactory
from halotools.empirical_models.ia_models.ia_model_components import CentralAlignment, RadialSatelliteAlignment
from halotools.empirical_models.ia_models.ia_strength_models import RadialSatelliteAlignmentStrength
from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens, Zheng07Sats, SubhaloPhaseSpace
from halotools.mock_observables import tpcf
from halotools.mock_observables.ia_correlations import ee_3d, ed_3d

import time

import warnings
warnings.filterwarnings("ignore")

def correlate(row):
    func, args, kwargs = row
    return func(*args, **kwargs)

def correlate_wrapper(model, rbins, halocat):
    gal_table = model.mock.galaxy_table
    coords = np.array( [ gal_table["x"], gal_table["y"], gal_table["z"] ] ).T
    orientations = np.array( [ gal_table["galaxy_axisA_x"], gal_table["galaxy_axisA_y"], gal_table["galaxy_axisA_z"] ] ).T

    func_params = [
            ( tpcf, (coords, rbins, coords), {"period":halocat.Lbox} ),
            ( ed_3d, (coords, orientations, coords, rbins), {"period":halocat.Lbox} ),
            ( ee_3d, (coords, orientations, coords, orientations, rbins), {"period":halocat.Lbox} ),
    ]

    with mp.Pool(processes=3) as pool:
        results = pool.map(correlate, func_params)
    
    return np.array(results)

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

def correlate_parallel(model, rbins, halocat, runs):
    """
    This function will test doing each run in series, and running all correlations in parallel within each run.
    This is what is currently done in the main code, and serves as a null test (of sorts). We know this works. Treat
    it as a baseline, especially for time. (i.e. this will not clash, so if I can make another method that doesn't
    clash and is faster, then I can be sure that it is a valid improvement)
    """
    results = np.zeros((runs, 3, len(rbins)-1))
    for i in range(runs):
        model.mock.populate()
        results[i] = correlate_wrapper(model, rbins, halocat)

    return results

def iterate_parallel(model, rbins, halocat, runs):
    """
    This function will test doing each run in parallel, and running all correlations in series within each run.
    """
    # Because model is NOT picleable, we need to iterate and populate to build a list of inputs for the pool
    # This is done with the parallel calculate anyway, so it may not be as bad as it appears. maybe a bit RAM heavy with large catalogs...
    rows = []
    for _ in range(runs):
        model.mock.populate()
        gal_table = model.mock.galaxy_table
        coords = np.array( [ gal_table["x"], gal_table["y"], gal_table["z"] ] ).T
        orientations = np.array( [ gal_table["galaxy_axisA_x"], gal_table["galaxy_axisA_y"], gal_table["galaxy_axisA_z"] ] ).T

        rows.append( (coords, orientations, rbins, halocat.Lbox) )

    with mp.Pool(processes=runs) as pool:
        results = pool.starmap(iterate_wrapper, rows)

    return np.array(results)

def compare_clash(correlations):
    """
    This function will compare the resulting correlations to ensure they are different. With seed=None,
    calling model.populate() should reshuffles some things. The correlations should be similar, but if they are
    exactly the same, then the parallel jobs are reading and populating the same model in memory.
    """
    clash_mat = np.zeros((len(correlations), len(correlations)))
    for i in range(len(correlations)):
        for j in range(len(correlations)):
            if i == j:
                continue
            clash_mat[i, j] = int( np.all(correlations[i] == correlations[j]) )
    return clash_mat

def mask_bad_halocat(halocat):
    bad_mask = (halocat.halo_table["halo_axisA_x"] == 0) & (halocat.halo_table["halo_axisA_y"] == 0) & (halocat.halo_table["halo_axisA_z"] == 0)
    bad_mask = bad_mask ^ np.ones(len(bad_mask), dtype=bool)
    halocat._halo_table = halocat.halo_table[ bad_mask ]

if __name__ == "__main__":
    inner_runs = 10
    catalog = "bolplanck"
    #catalog = "multidark"
    # Set rbins - Larger max distance means longer run time (from correlation calculations)
    rbins = np.logspace(-1,1.2,21)
    rbin_centers = (rbins[:-1]+rbins[1:])/2.0

    ############################################################################################################################

    # Initial strength parameters
    # These don't matter, as they are only needed for the initial creation of the model
    central_alignment_strength = 0.8
    satellite_alignment_strength = 0.8

    # Satellite bins
    sat_bins = np.logspace(10.5, 15.2, 15)

    # Set up halocat
    halocat = CachedHaloCatalog(simname=catalog, halo_finder='rockstar', redshift=0, version_name='halotools_v0p4')
    mask_bad_halocat(halocat)
    
    cens_occ_model = Zheng07Cens()
    cens_prof_model = TrivialPhaseSpace()
    cens_orientation = CentralAlignment(central_alignment_strength=central_alignment_strength)

    sats_occ_model = Zheng07Sats()
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
    model.populate_mock(halocat, seed=None)

    print("\nStarting parallel correlations within series loops")
    start = time.time()
    correlations_A = correlate_parallel(model, rbins, halocat, inner_runs)
    end = time.time()
    print("Time taken for parallel correlations within series loops: ", end-start)

    clash_mat = compare_clash(correlations_A[:,0,:])          # Only look at pos-pos correlation
    print("Clas Matrix shape: ", clash_mat.shape)
    print("Clashing: ", np.any(clash_mat))

    print("\n>>>>>\n")

    print("Starting series correlations within parallel loops")
    start = time.time()
    correlations_B = iterate_parallel(model, rbins, halocat, inner_runs)
    end = time.time()
    print("Time taken for series correlations within parallel loops: ", end-start)

    clash_mat = compare_clash(correlations_B[:,0,:])          # Only look at pos-pos correlation
    print("Clash Matrix shape: ", clash_mat.shape)
    print("Clashing: ", np.any(clash_mat))

    # Visualize the results
    # Three panel plot, one for each type of correlation
    # Plot all values from the parallel correlations set in blue and the parallel iterate set in red
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    for i in range(3):
        for j in range(inner_runs):
            ax[i].plot(rbin_centers, correlations_A[j,i,:], color='blue', alpha=0.1)
            ax[i].plot(rbin_centers, correlations_B[j,i,:], color='red', alpha=0.1)
        ax[i].set_xscale('log')
        if i != 2:
            ax[i].set_yscale('log')
        ax[i].set_xlabel("Distance (Mpc/h)")

    plt.show()

    # From testing, time for parallel correlations follow
    # t = 1.26x - 0.27
    # And time for parallel loops follows
    # t = 0.45x + 1.88
    # where x is the number of runs
    # Parallelizing the loops scales much better (not surprising, we're basically just saying that more
    # resources thrown at the problem makes it better if you use all the resources efficiently)
    # The setup loop that fills lists with the required positions and orientations does not appreciably increase time