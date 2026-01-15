default_params = ['mass_1_source', 'mass_2_source', 'mass_ratio', 'chirp_mass_source', 'redshift', 'luminosity_distance', 'chi_eff']
key = 'MDC/posterior_samples'

from astropy.cosmology import Planck15
import astropy.units as u
import astropy.constants as const

def ddL_dz(z):

    dc = Planck15.comoving_distance(z) + const.c * (1. + z) / Planck15.H(z)
    return dc.to(u.Mpc).value

if __name__ == '__main__':

    import argparse
    import os
    import h5py
    import numpy as np

    import sys
    sys.path.append('/work/aqc/lib/effective-spin-priors')
    from priors import chi_effective_prior_from_isotropic_spins

    parser = argparse.ArgumentParser()
    parser.add_argument('-params', '--params', type=str, nargs='+', default=default_params,
                        help='Parameters to get samples for.')
    parser.add_argument('-in_path', '--in_path', type=str,
                        help='Path to directory with PE samples')
    parser.add_argument('-out_path', '--out_path', type=str, default=None,
                        help='Output directory')
    parser.add_argument('-target_nsamp', '--target_nsamp', type=int, default=6657,
                        help='Target number of PE samples to save')
    args = parser.parse_args()

    dict_keys = args.params + ['prior']
    priors, PE_samples = {param: [] for param in dict_keys}, {param: [] for param in dict_keys}

    os.makedirs(args.out_path, exist_ok=True)

    for fn in sorted(os.listdir(args.in_path)):
        if fn.endswith('.h5'):
            with h5py.File(os.path.join(args.in_path, fn), 'r') as f:

                data = f[key]

                nsamps_tot = data['chi_eff'].size
                if nsamps_tot > args.target_nsamp:
                    inds = np.sort(np.random.choice(nsamps_tot, args.target_nsamp, replace=False))
                else:
                    inds = np.arange(nsamps_tot)

                for param in args.params:
                    PE_samples[param].append(data[param][inds])
                
                # compute prior -- p(m1, q, chieff, z)
                # sylvia's MDC used flat in Mc and q, aligned spin uniform with aMax=0.8, dL^2 prior 
                dL, z, q, chieff = PE_samples['luminosity_distance'][-1], PE_samples['redshift'][-1], PE_samples['mass_ratio'][-1], PE_samples['chi_eff'][-1]
                eta = q / (1. + q)**2 

                prior = dL**2 # P(Mc, q, dL)
                prior *= eta**(3/5) * (1. + q) # P(Mc, q) -> P(m1, q)
                prior *= chi_effective_prior_from_isotropic_spins(chieff, q, aMax=0.8) #P(chieff)
                prior *= ddL_dz(z) # P(dL) -> P(z)

                PE_samples['prior'].append(prior)

    # stack samples
    # get minimum n samples 
    nsamps = min([len(i) for i in PE_samples[args.params[0]]])
    print(f'Using {nsamps} samples for each event (target: {args.target_nsamp})')

    for k in dict_keys:
        PE_samples[k] = np.vstack([i[:nsamps] for i in PE_samples[k]])

    # save 
    np.savez(os.path.join(args.out_path, 'MDC_PE_samples.npz'), **PE_samples)