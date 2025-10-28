import h5py
import numpy as np
import os

import astropy.units as u
from astropy.cosmology import Planck15

injpath = '/work/aqc/data/GWTC_data/LVK_injections/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5'
injoutdir = '/work/aqc/data/pymcpop/PE_samples/GWTC-injs'
prefix = ''

far_thr = 1 #1/yr
rho_thr = 10

if __name__ == '__main__':
    print(f'Processing injections from {injpath}')
    with h5py.File(injpath, 'r') as f_:
        f = f_['injections']

        injs = np.vstack([f['mass1_source'][()], f['mass2_source'][()], f['redshift'][()], 
                          f['spin1x'][()], f['spin1y'][()], f['spin1z'][()], 
                          f['spin2x'][()], f['spin2y'][()], f['spin2z'][()], 
                          f['sampling_pdf'][()], f['mixture_weight'][()]]).T
        far_min = np.min([item[()] for key, item in f.items() if key.startswith('far')], axis=0)
        rho_opt = f['optimal_snr_net'][()]
        run = np.array([int(str(i[-1])) for i in np.char.decode(f['name'][()])])
        mask = np.where(run==3, far_min <= far_thr, rho_opt >= rho_thr)

        injs = injs[mask] # save only detected events

        # transform into m1det, m2det, dL
        z = injs[:,2]
        m1d = injs[:,0] * (1 + z)
        m2d = injs[:,1] * (1 + z)
        dL = Planck15.luminosity_distance(z).to(u.Mpc).value

        log_p_draw = np.log(injs[:,-2]) 
        log_mix_wts = np.log(injs[:,-1])

        # save everything
        if not os.path.exists(injoutdir):
            os.makedirs(injoutdir)
        os.chdir(injoutdir)

        np.save(prefix+'Ngen.npy', np.asarray(int(f.attrs['total_generated'])))
        np.save(prefix+'Tobs.npy', np.asarray(f.attrs['analysis_time_s']))

        np.save(prefix+'m1d.npy', m1d)
        np.save(prefix+'m2d.npy', m2d)
        np.save(prefix+'dL.npy', dL)
        np.save(prefix+'log_p_draw.npy', log_p_draw)
        np.save(prefix+'log_mix_wts.npy', log_mix_wts)

        for i, spin in enumerate(['spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z']):
            np.save(prefix+f'{spin}.npy', injs[:,i+3])
