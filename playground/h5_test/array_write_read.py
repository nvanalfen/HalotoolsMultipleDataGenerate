from halotools.sim_manager import CachedHaloCatalog

from halotools.empirical_models import HodModelFactory
from halotools.empirical_models.ia_models.ia_model_components import CentralAlignment, RadialSatelliteAlignment
from halotools.empirical_models.ia_models.ia_strength_models import RadialSatelliteAlignmentStrength
from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens, Zheng07Sats, SubhaloPhaseSpace

import numpy as np
import time
import h5py
import os
# Change the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import warnings
warnings.filterwarnings("ignore")

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
    cens_occ_model.param_dict["logMmin"] = 11.0
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

def write_basic_h5(table, columns, file_name):
    with h5py.File(file_name, "w") as f:
        f.create_dataset("table", data=table[columns])

def write_clever_h5(table, columns, file_name):
    with h5py.File(file_name, "w") as f:
        for col in columns:
            f.create_dataset(col, data=table[col])

def read_basic_h5(file_name, columns):
    start = time.time()
    table = []
    with h5py.File(file_name, "r") as f:
        table = np.vstack([ f["table"][:][col] for col in columns ]).T
    end = time.time()
    return end-start, table

def read_clever_h5(file_name, columns):
    start = time.time()
    table = []
    with h5py.File(file_name, "r") as f:
        for col in columns:
            table.append(f[col][:])
    table = np.vstack(table).T
    end = time.time()
    return end-start, table

if __name__ == "__main__":
    halocat = CachedHaloCatalog(simname="bolplanck", redshift=0.0,
                                halo_finder="rockstar", version_name="halotools_v0p4")
    mask_bad_halocat(halocat)
    sat_bins = np.logspace(10.5, 15.2, 15)

    model = build_model_instance(cen_strength=1.0,sat_params=1.0, sat_bins=sat_bins, halocat=halocat, constant=True)
    print(len(model.mock.galaxy_table))

    columns = ["x", "y", "z", "halo_axisA_x", "halo_axisA_y", "halo_axisA_z"]
    table = model.mock.galaxy_table

    write_basic_h5(table, columns, "basic.h5")
    write_clever_h5(table, columns, "clever.h5")

    print("Read basic h5")
    time_basic, table_basic = read_basic_h5("basic.h5", columns)
    print(f"Time taken: {time_basic:.4f} seconds")
    print(f"Shape: {table_basic.shape}")
    print(table_basic[:5])

    print()

    print("Read clever h5")
    time_clever, table_clever = read_clever_h5("clever.h5", columns)
    print(f"Time taken: {time_clever:.4f} seconds")
    print(f"Shape: {table_clever.shape}")
    print(table_clever[:5])

    print("Are the tables equal?", np.array_equal(table_basic, table_clever))