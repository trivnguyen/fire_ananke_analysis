#!/usr/bin/env python
# coding: utf-8

import os
import h5py
import argparse

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18  # Planck 2018

import agama
import gizmo_analysis as ga
import utilities as ut

FLAGS = None

def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', required=True, help='Path to output file')
    parser.add_argument('-gal', required=True, help='FIRE Galaxy')
    parser.add_argument('-ijob', required=True, type=int, help='Index of job')
    parser.add_argument('-Njob', required=True, type=int, help='Total number of jobs')
    return parser.parse_args()


if __name__ == '__main__':

    FLAGS = parse_cmd()

    # Parameters
    sim_dir = f'/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/{FLAGS.gal}'
    nsnap = 600
    min_int_time = 7   # minimum integration time in Gyr
    trajsize = 100

    # Reading in galaxy potential with Agama
    DM_pot_file = f'potentials/{FLAGS.gal}/600.dark.axi_4.coef_mul_DR'   # DM halo + cold gas (multiple)
    bar_pot_file = f'potentials/{FLAGS.gal}/600.bar.axi_4.coef_cylsp_DR' # bar + hot gas (cubicspline)

    print(f'Reading DM potential from {DM_pot_file}')
    print(f'Reading bar potential from {bar_pot_file}')
    pxr_DM = agama.Potential(file=DM_pot_file)
    pxr_bar = agama.Potential(file=bar_pot_file)

    print('Files loaded!')
    potential = agama.Potential(pxr_DM, pxr_bar)

    # Read in IDs and redshifts of accreted stars
    table = pd.read_csv(f'accretion_history/stars_accretion_history_{FLAGS.gal}_v2.csv')
    idx = table['id_stars'].values
    zacc = table['redshift_stars'] .values

    # estimate the lookback time (in Gyr) for all redshifts
    # this will be our integration time
    time = -Planck18.lookback_time(zacc).value  # negative because integrate backward in time
    time = np.minimum(time, -min_int_time)

    # Get job index and apply cut to IDs
    N_samples = len(idx)
    i_start = N_samples * FLAGS.ijob // FLAGS.Njob
    i_stop = N_samples * (FLAGS.ijob + 1) // FLAGS.Njob
    idx = idx[i_start: i_stop]
    time = time[i_start: i_stop]

    print(f'Total number of samples: {N_samples}')
    print(f'Job  : {FLAGS.ijob} / {FLAGS.Njob}')
    print(f'Index: [{i_start}, {i_stop}]')

    # Read in snapshot using gizmo
    # and get position and velocities of all accreted stars
    part = ga.io.Read.read_snapshots(
        species=['star'], snapshot_values=nsnap,
        simulation_directory=sim_dir,
        properties=['Coordinates', 'Velocities'],
        assign_hosts=True,
        assign_hosts_rotation=True)
    pos = part['star'].prop("host.distance.principal")[idx]
    vel = part['star'].prop("host.velocity.principal")[idx]

    pos_vel = np.hstack([pos, vel])

    # Start integrate orbit by batch
    peris = []
    apos = []
    orbit = agama.orbit(
        ic=pos_vel, time=time, potential=potential, trajsize=trajsize)
    for i in range(len(orbit)):
        r = np.linalg.norm(orbit[i][1][:, :3], axis=1)
        peris.append(np.min(r))
        apos.append(np.max(r))
    peris = np.array(peris)
    apos = np.array(apos)

    # Save to HDF5 file
    print(f'Writing output to: {FLAGS.output}')
    with h5py.File(FLAGS.output, 'w') as f:
        f.attrs.update({
            'num_samples': N_samples,
            'i_job': FLAGS.ijob,
            'N_job': FLAGS.Njob,
            'i_start': i_start,
            'i_stop': i_stop,
        })
        f.create_dataset('pericenters', data=peris, dtype=np.float32)
        f.create_dataset('apocenters', data=apos, dtype=np.float32)
        f.create_dataset('id_stars', data=idx, dtype=np.int32)

