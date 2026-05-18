# python -u process_GW241011_GW241110_PE.py --out_path /work/aqc/data/GWTC_data/processed

import sys

sys.path.append(
    "/work/aqc/lib/effective-spin-priors"
)  # replace with your path to effective-spin-priors

DATA_DIR = "/work/aqc/data/GWTC_data/GW241011_GW241110_data_release/data/standard_pe"

EVENTS = {
    "GW241011": f"{DATA_DIR}/GW241011_posterior_samples_combined.h5",
    "GW241110": f"{DATA_DIR}/GW241110_posterior_samples_combined.h5",
}

default_params = [
    "mass_1_source",
    "mass_2_source",
    "mass_ratio",
    "chirp_mass_source",
    "redshift",
    "chi_eff",
]

if __name__ == "__main__":
    import argparse
    import os

    import h5py
    import numpy as np
    from astropy.cosmology import Planck15
    from priors import chi_effective_prior_from_isotropic_spins

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-params",
        "--params",
        type=str,
        nargs="+",
        default=default_params,
        help="Parameters to get samples for.",
    )
    parser.add_argument("-out_path", "--out_path", type=str, required=True, help="Output directory")
    parser.add_argument(
        "-target_nsamp",
        "--target_nsamp",
        type=int,
        default=9000,
        help="Target number of PE samples to save",
    )
    args = parser.parse_args()

    PE_samples = {param: [] for param in args.params}
    wf_per_event = []

    if args.out_path is not None:
        os.makedirs(args.out_path, exist_ok=True)

    for event_name, filepath in EVENTS.items():
        print(f"Processing {event_name}")
        with h5py.File(filepath, "r") as f:
            key = [k for k in f.keys() if "IMRPhenomXPHM" in k][0]
            wf_per_event.append([event_name, key.split(":")[-1]])

            ps = f[key]["posterior_samples"]
            nsamps_tot = ps.size
            if nsamps_tot > args.target_nsamp:
                inds = np.sort(np.random.choice(nsamps_tot, args.target_nsamp, replace=False))
            else:
                inds = np.arange(nsamps_tot)

            for param in args.params:
                PE_samples[param].append(ps[param][inds])

    # get minimum n samples across events and downsample to match
    nsamps = min(len(s) for s in PE_samples[args.params[0]])
    print(f"Using {nsamps} samples for each event (target: {args.target_nsamp})")

    for event_idx in range(len(PE_samples[args.params[0]])):
        inds = np.random.randint(0, len(PE_samples[args.params[0]][event_idx]), nsamps)
        for param in args.params:
            PE_samples[param][event_idx] = PE_samples[param][event_idx][inds]

    for param in args.params:
        PE_samples[param] = np.vstack(PE_samples[param])

    # prior: uniform in det-frame masses, uniform in comoving volume, isotropic spins
    # p(m1, q, chieff, z)
    z = PE_samples["redshift"]
    m1 = PE_samples["mass_1_source"]
    q = PE_samples["mass_ratio"]
    chi_eff = PE_samples["chi_eff"]

    prior = (1 + z) ** 2 * Planck15.differential_comoving_volume(z).value / 1e9  # p(m1,m2,z)
    prior *= m1  # p(m1,m2) -> p(m1, q)
    prior *= chi_effective_prior_from_isotropic_spins(chi_eff=chi_eff, q=q, aMax=0.99)

    PE_samples["prior"] = prior

    prefix = "GW241011_GW241110"

    np.savez(os.path.join(args.out_path, prefix + "_PE_samples.npz"), **PE_samples)
    np.savetxt(os.path.join(args.out_path, prefix + "_waveforms.txt"), wf_per_event, fmt="%s")
    print(f"Saved to {args.out_path}/{prefix}_PE_samples.npz")
