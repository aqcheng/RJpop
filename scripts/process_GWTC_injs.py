import sys

import h5py
import numpy as np

sys.path.append('/work/aqc/lib/effective-spin-priors')  # replace with your path to effective-spin-priors
from priors import chi_effective_prior_from_isotropic_spins

inpath = '/work/aqc/data/GWTC_data/LVK_injections/mixture-semi_o1_o2-real_o3_o4a-polar_spins_20250503134659UTC.hdf'  # replace with your path
outpath = '/work/aqc/data/GWTC_data/processed/o1_o2_o3_o4a_injs.npz'  # replace with your desired output path

far_thr = 1 #1/yr
rho_thr = 10

if __name__ == '__main__':
    print(f'Processing injections from {inpath}')
    with h5py.File(inpath, 'r') as f_:
        f = f_['events']

        injs = {}

        m1, m2 = f['mass1_source'][()], f['mass2_source'][()]
        q = m2 / m1
        chirp_mass = (m1 * m2) ** (3/5) / (m1 + m2) ** (1/5)

        injs['mass_1_source'] = m1
        injs['mass_2_source'] = m2
        injs['mass_ratio'] = q
        injs['chirp_mass_source'] = chirp_mass

        injs['redshift'] = f['redshift'][()]

        theta1, theta2 = f['spin1_polar_angle'][()], f['spin2_polar_angle'][()]
        a1z = f['spin1_magnitude'][()] * np.cos(theta1)
        a2z = f['spin2_magnitude'][()] * np.cos(theta2)
        chi_eff = (m1 * a1z + m2 * a2z) / (m1 + m2)

        injs['chi_eff'] = chi_eff

        sampling_pdf = np.exp(f['lnpdraw_mass1_source_mass2_source_redshift_spin1_magnitude_spin1_polar_angle_spin1_azimuthal_angle_spin2_magnitude_spin2_polar_angle_spin2_azimuthal_angle'][()])
        sampling_pdf *= m1 # m1, m2 -> m1, q 
        sampling_pdf /= (1/(4*np.pi))**2 * np.sin(theta1) * np.sin(theta2) # uniform prior on spins
        sampling_pdf *= chi_effective_prior_from_isotropic_spins(chi_eff=chi_eff, q=q, aMax=1.0)
        injs['prior'] = sampling_pdf

        injs['w'] = f['weights'][()]

        far_min = np.min([f[name][()] for name in f.dtype.names if name.endswith('_far')], axis=0)
        rho_opt = f['semianalytic_observed_phase_maximized_snr_net'][()]

        injs['far'] = far_min
        injs['rho'] = rho_opt

        mask = (rho_opt >= rho_thr) | (far_min <= far_thr)

        injs = {k : v[mask] for k, v in injs.items()} # save only detected events

        injs['total_generated'] = f_.attrs['total_generated']
        injs['Tobs_yr'] = f_.attrs['total_analysis_time'] / 3.1557e7 # in years

        # save everything
        np.savez(outpath, **injs)
        print('Saved the following keys: ', injs.keys())
