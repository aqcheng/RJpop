import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rjpop'))
from rjpop.effective_spin_priors import chi_effective_prior_from_isotropic_spins
from rjpop.load_config import load_config

FAR_THR = 1   # 1/yr
RHO_THR = 11

if __name__ == '__main__':
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description='Pre-process combined injection campaign, assuming polar spins.'
    )
    parser.add_argument(
        '--in_path', '-in_path', type=str,
        default=cfg.get('gwtc_injs_in_path'),
        help='Path to the injection HDF file. '
             'Default: gwtc_injs_in_path in ~/.rjpop_config.json',
    )
    parser.add_argument(
        '--out_path', '-out_path', type=str,
        default=cfg.get('lvk_out_path'),
        help='Output directory. Default: out_path in ~/.rjpop_config.json',
    )
    parser.add_argument(
        '--far_thr', type=float, default=FAR_THR,
        help=f'FAR threshold in 1/yr for detection cut (default: {FAR_THR})',
    )
    parser.add_argument(
        '--rho_thr', type=float, default=RHO_THR,
        help=f'Optimal SNR threshold for detection cut (default: {RHO_THR})',
    )
    args = parser.parse_args()

    if args.in_path is None:
        parser.error("--in_path is required (or set 'gwtc_injs_in_path' in ~/.rjpop_config.json)")
    if args.out_path is None:
        parser.error("--out_path is required (or set 'out_path' in ~/.rjpop_config.json)")

    os.makedirs(args.out_path, exist_ok=True)

    # get prefix from .md file
    prefix = ""
    for fn in os.listdir(os.path.dirname(args.in_path)):
        if fn.endswith('.md'):
            prefix = fn.split("_")[1] + "_"
            break
    
    outfile = os.path.join(args.out_path, f'{prefix}injs.npz')

    print(f'Processing injections from {args.in_path}')
    with h5py.File(args.in_path, 'r') as f_:
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
        sampling_pdf *= m1  # m1, m2 -> m1, q
        sampling_pdf /= (1/(4*np.pi))**2 * np.sin(theta1) * np.sin(theta2)  # uniform prior on spins
        sampling_pdf *= chi_effective_prior_from_isotropic_spins(chi_eff=chi_eff, q=q, aMax=1.0)
        injs['prior'] = sampling_pdf

        injs['w'] = f['weights'][()]

        far_min = np.min([f[name][()] for name in f.dtype.names if name.endswith('_far')], axis=0)
        rho_opt = f['semianalytic_observed_phase_maximized_snr_net'][()]

        injs['far'] = far_min
        injs['rho'] = rho_opt

        # use SNR threshold only when FAR not available, i.e. O1 and O2
        mask = ((rho_opt >= args.rho_thr) & ~np.isfinite(far_min)) | (far_min <= args.far_thr)

        injs = {k: v[mask] for k, v in injs.items()}  # save only detected events

        injs['total_generated'] = f_.attrs['total_generated']
        injs['Tobs_yr'] = f_.attrs['total_analysis_time'] / 3.1557e7  # in years

        np.savez(outfile, **injs)
        print(f'Saved to {outfile}')
        print('Keys: ', list(injs.keys()))
