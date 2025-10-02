# -------------------------
# ----> GWTC3 helpers <----
# -------------------------

GWTC3_BBHs = """
GW150914_095045
GW151012_095443
GW151226_033853
GW170104_101158
GW170608_020116
GW170729_185629
GW170809_082821
GW170814_103043
GW170818_022509
GW170823_131358
GW190408_181802
GW190412_053044
GW190413_134308
GW190421_213856
GW190503_185404
GW190512_180714
GW190513_205428
GW190517_055101
GW190519_153544
GW190521_030229
GW190521_074359
GW190527_092055
GW190602_175927
GW190620_030421
GW190630_185205
GW190701_203306
GW190706_222641
GW190707_093326
GW190708_232457
GW190720_000836
GW190727_060333
GW190728_064510
GW190803_022701
GW190828_063405
GW190828_065509
GW190910_112807
GW190915_235702
GW190924_021846
GW190925_232845
GW190929_012149
GW190930_133541
GW191105_143521
GW191109_010717
GW191127_050227
GW191129_134029
GW191204_171526
GW191215_223052
GW191216_213338
GW191222_033537
GW191230_180458
GW200112_155838
GW200128_022011
GW200129_065458
GW200202_154313
GW200208_130117
GW200209_085452
GW200219_094415
GW200224_222234
GW200225_060421
GW200302_015811
GW200311_115853
GW200316_215756
GW190413_052954
GW190719_215514
GW190725_174728
GW190731_140936
GW190805_211137
GW191103_012549
GW200216_220804
""".split()

GWTC4_BBHs = """
GW230601_224134
GW230605_065343
GW230606_004305
GW230608_205047
GW230609_064958
GW230624_113103
GW230627_015337
GW230628_231200
GW230630_070659∗
GW230630_125806
GW230630_234532
GW230702_185453
GW230704_021211
GW230704_212616
GW230706_104333
GW230707_124047
GW230708_053705
GW230708_230935
GW230709_122727
GW230712_090405
GW230723_101834
GW230726_002940
GW230729_082317
GW230731_215307
GW230803_033412
GW230805_034249
GW230806_204041
GW230811_032116
GW230814_061920
GW230814_230901
GW230819_171910
GW230820_212515
GW230824_033047
GW230825_041334
GW230831_015414
GW230904_051013
GW230911_195324
GW230914_111401
GW230919_215712
GW230920_071124
GW230922_020344
GW230922_040658
GW230924_124453
GW230927_043729
GW230927_153832
GW230928_215827
GW230930_110730
GW231001_140220
GW231004_232346
GW231005_021030
GW231005_091549
GW231008_142521
GW231014_040532
GW231018_233037
GW231020_142947
GW231028_153006
GW231029_111508
GW231102_071736
GW231104_133418
GW231108_125142
GW231110_040320
GW231113_122623
GW231113_200417
GW231114_043211
GW231118_005626
GW231118_071402
GW231118_090602
GW231119_075248
GW231123_135430
GW231127_165300
GW231129_081745
GW231206_233134
GW231206_233901
GW231213_111417
GW231221_135041
GW231223_032836
GW231223_075055
GW231223_202619
GW231224_024321
GW231226_101520
GW231230_170116
GW231231_154016
GW240104_164932
GW240107_013215
GW240109_050431
""" # exclude NSBHs GW230518_125908 and GW230529_181500

default_params = ['mass_1_source', 'mass_2_source', 'mass_ratio', 'chirp_mass_source', 'redshift', 'chi_eff']

if __name__ == '__main__':

    import argparse
    import os
    import re
    import h5py
    import numpy as np
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-params', '--params', type=str, nargs='+', default=default_params,
                        help='Parameters to get samples for.')
    parser.add_argument('-in_path', '--in_path', type=str, nargs='+',
                        help='Path(s) to directories with LVK PE samples')
    parser.add_argument('-out_path', '--out_path', type=str, default=None,
                        help='Output directory')
    parser.add_argument('-catalogs', '--catalogs', type=str, nargs='+', choices=['GWTC2p1', 'GWTC3', 'GWTC4'],
                        default=['GWTC2p1', 'GWTC3', 'GWTC4'], help='Catalogs to use')
    parser.add_argument('-target_nsamp', '--target_nsamp', type=int, default=5000,
                        help='Target number of PE samples to save')
    args = parser.parse_args()

    priors, PE_samples = {param: [] for param in args.params}, {param: [] for param in args.params}

    os.makedirs(args.out_path, exist_ok=True)

    for catname in sorted(args.catalogs):
        print(f'Processing catalog {catname}')
        if catname == 'GWTC4':
            eventnames = GWTC4_BBHs
            fn_suffix = '.hdf5'
        else:
            eventnames = GWTC3_BBHs
            fn_suffix = '_cosmo.h5'
        
        prior_saved = False
        
        for path in args.in_path:
            # look for files corresponding to the catalog
            for fn in sorted(os.listdir(path)):
                if fn.startswith(f'IGWN-{catname}') and 'PEDataRelease' in fn and \
                   fn.endswith(fn_suffix):

                    event_name = 'GW' + '_'.join(re.split('[_-]+', fn.split('-GW')[2])[:2])
                    if event_name in eventnames:
                        print(f'    {event_name}')
                        with h5py.File(os.path.join(path, fn), 'r') as f:

                            mixed_key = [k for k in f.keys() if 'Mixed' in k][0]
                            
                            nsamps_tot = f[mixed_key]['posterior_samples'].size
                            if nsamps_tot > args.target_nsamp:
                                inds = np.sort(np.random.choice(nsamps_tot, args.target_nsamp, replace=False))
                            else:
                                inds = np.arange(nsamps_tot)

                            for param in args.params:
                                PE_samples[param].append(f[mixed_key]['posterior_samples'][inds, param])

                            if not prior_saved:
                                phenom_key = [k for k in f.keys() if 'IMRPhenom' in k][0]
                                try: # save a prior dictionary
                                    priordict = {key : f[phenom_key]['priors']['analytic'][key][0].decode('utf8') for key in f[phenom_key]['priors']['analytic']}
                                    if priordict:
                                        prior_fn = os.path.join(args.out_path, f'{catname}_prior.json')
                                        with open(prior_fn, 'w') as pf:
                                            json.dump(priordict, pf)
                                        prior_saved = True
                                        print(f'    {catname} prior saved from event {event_name}')
                                except Exception as e:
                                    pass

    # stack samples
    # get minimum n samples 
    nsamps = min([len(i) for i in PE_samples[args.params[0]]])
    print(f'Using {nsamps} samples for each event (target: {args.target_nsamp})')

    for param in args.params:
        PE_samples[param] = np.vstack([i[:nsamps] for i in PE_samples[param]])
    # save 
    np.savez(os.path.join(args.out_path, 'GWTC_PE_samples.npz'), **PE_samples)