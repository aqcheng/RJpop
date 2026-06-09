import h5py
import numpy as np

from rjpop.effective_spin_priors import chi_effective_prior_from_isotropic_spins

inpath = '/work/aqc/data/GWTC_data/LVK_injections/endo3_bbhpop-LIGO-T2100113-v12.hdf5'  # replace with your path
outpath = '/work/aqc/data/GWTC_data/processed/o3_injs.npz'  # replace with your desired output path
prefix = ''

# far_thr = 1 #1/yr
rho_thr = 9

if __name__ == '__main__':
    print(f'Processing injections from {inpath}')
    with h5py.File(inpath, 'r') as f_:
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

        a1x, a2x = f['spin1x'][()], f['spin2x'][()]
        a1y, a2y = f['spin1y'][()], f['spin2y'][()]
        a1z, a2z = f['spin1z'][()], f['spin2z'][()]
        chi_eff = (m1 * a1z + m2 * a2z) / (m1 + m2)

        injs['chi_eff'] = chi_eff

        sampling_pdf = f['mass1_source_mass2_source_sampling_pdf'][()] * \
                       f['redshift_sampling_pdf'][()]  # this is P(m1, m2, z) -- want to convert to P(m1, q, z, chi_eff)
        sampling_pdf *= m1 # m1, m2 -> m1, q
        sampling_pdf *= chi_effective_prior_from_isotropic_spins(chi_eff=chi_eff, q=q, aMax=0.998)
        injs['prior'] = sampling_pdf

        injs['w'] = f['mixture_weight'][()]

        far_min = np.min([item[()] for key, item in f.items() if key.startswith('far')], axis=0)
        rho_opt = f['optimal_snr_net'][()]

        injs['far'] = far_min
        injs['rho'] = rho_opt

        # run = np.array([int(str(i[-1])) for i in np.char.decode(f['name'][()])])
        # mask = np.where(run==3, far_min <= far_thr, rho_opt >= rho_thr)
        # I'm running on Sylvia's injections -- just do SNR cutoff
        mask = rho_opt >= rho_thr

        injs = {k : v[mask] for k, v in injs.items()} # save only detected events

        injs['total_generated'] = f_.attrs['total_generated']
        injs['Tobs_yr'] = f_.attrs['analysis_time_s'] / 3.1557e7 # in years

        # save everything
        np.savez(outpath, **injs)
        print('Saved the following keys: ', injs.keys())
