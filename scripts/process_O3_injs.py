import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rjpop'))
from rjpop.effective_spin_priors import chi_effective_prior_from_isotropic_spins
from rjpop.load_config import load_config

RHO_THR = 9

if __name__ == '__main__':
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description='Preprocess O3-only injection campaign into NPZ format.'
    )
    parser.add_argument(
        '--in_path', '-in_path', type=str,
        default=cfg.get('o3_injs_in_path'),
        help='Path to the O3 injection HDF file. '
             'Default: o3_injs_in_path in ~/.rjpop_config.json',
    )
    parser.add_argument(
        '--out_path', '-out_path', type=str,
        default=cfg.get('out_path'),
        help='Output directory. Default: out_path in ~/.rjpop_config.json',
    )
    parser.add_argument(
        '--rho_thr', type=float, default=RHO_THR,
        help=f'Optimal SNR threshold for detection cut (default: {RHO_THR})',
    )
    args = parser.parse_args()

    if args.in_path is None:
        parser.error("--in_path is required (or set 'o3_injs_in_path' in ~/.rjpop_config.json)")
    if args.out_path is None:
        parser.error("--out_path is required (or set 'out_path' in ~/.rjpop_config.json)")

    os.makedirs(args.out_path, exist_ok=True)
    outfile = os.path.join(args.out_path, 'o3_injs.npz')

    print(f'Processing injections from {args.in_path}')
    with h5py.File(args.in_path, 'r') as f_:
        f = f_['injections']

        injs = {}

        m1, m2 = f['mass1_source'][()], f['mass2_source'][()]
        q = m2 / m1
        chirp_mass = (m1 * m2) ** (3/5) / (m1 + m2) ** (1/5)

        injs['mass_1_source'] = m1
        injs['mass_2_source'] = m2
        injs['mass_ratio'] = q
        injs['chirp_mass_source'] = chirp_mass

        injs['redshift'] = f['redshift'][()]

        a1z, a2z = f['spin1z'][()], f['spin2z'][()]
        chi_eff = (m1 * a1z + m2 * a2z) / (m1 + m2)

        injs['chi_eff'] = chi_eff

        sampling_pdf = (
            f['mass1_source_mass2_source_sampling_pdf'][()]
            * f['redshift_sampling_pdf'][()]  # P(m1, m2, z) -> convert to P(m1, q, z, chi_eff)
        )
        sampling_pdf *= m1  # m1, m2 -> m1, q
        sampling_pdf *= chi_effective_prior_from_isotropic_spins(chi_eff=chi_eff, q=q, aMax=0.998)
        injs['prior'] = sampling_pdf

        injs['w'] = f['mixture_weight'][()]

        far_min = np.min([item[()] for key, item in f.items() if key.startswith('far')], axis=0)
        rho_opt = f['optimal_snr_net'][()]

        injs['far'] = far_min
        injs['rho'] = rho_opt

        mask = rho_opt >= args.rho_thr

        injs = {k: v[mask] for k, v in injs.items()}  # save only detected events

        injs['total_generated'] = f_.attrs['total_generated']
        injs['Tobs_yr'] = f_.attrs['analysis_time_s'] / 3.1557e7  # in years

        np.savez(outfile, **injs)
        print(f'Saved to {outfile}')
        print('Keys: ', list(injs.keys()))
