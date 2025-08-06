#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import plotting_related_functions
import KC_population_calcium_rate_model_functions as mdl
import os
import pandas as pd
from tqdm import tqdm
import itertools as it

rng = np.random.default_rng(420) # Set random seed

extensions = ['png', 'pdf'] # Figure save extensions
# Initialize model parameter values
# ----------------------------------

n_inps = 5000 # Number of different input scaling cases for the model to average / whatever over xD
chunksize = 50  # The chunk size to run in parallel for the model.


# PARAMETERS RELEVANT FOR FURTHER ANALYSIS MIMICKING the sampling in calcium imaging
# 1) NOISE
nstd = 0.05 # noise standard deviation
nmean = 0.0 # noise mean
# 2) AVERAGING OVER KCs
nKC_per_avg = 100 # Number of KCs to average over per each simulation


# Simulation save directory
sdir = r'../data/model_related/odor_activity_dists'
if not os.path.exists(sdir):
    os.makedirs(sdir)
sname = f'baseline_and_steady_odor_response_n_iter_{n_inps}.npz' # Simulation save name


if sname in os.listdir(sdir):
    print('Simulation with the given number of inputs is already run, loading previous results & skipping to figure generation.')
    loader = np.load(os.path.join(sdir,sname), allow_pickle=True)
    resps = loader['resps']
    blines = loader['blines']
    nKCs = np.repeat(loader['modelparams'],2)[0]['nKCs']
    inhfactor = np.repeat(loader['modelparams'],2)[0]['inhscale']*nKCs

else:
    print(f'Running {n_inps} simulations for the activity distribution.')
    # Load the fit parameters
    fitparams = np.load(r'../data/model_related/fit_results/model_fit_normfac_3.885.npz')
    tauKCdec, tauinp, tauadapt, adaptscale, bline, n_str = fitparams['fittedpars_KD']
    tauinh, inhfactor, infp, slf = fitparams['fittedpars_WT']
    inhfactor_oe = float(fitparams['inhfactor_OE'])
    nKCs, prr, pur, __ = fitparams['restparams']
    nKCs = int(nKCs)

    # Remaining parameters not fitted to any data (mostly learning-related)
    dt = 0.01  # time step this should be sufficient actually
    # Adjust noise strength to time constant
    n_str = n_str * np.sqrt(2 / tauKCdec * dt)
    inhscale = np.ones([nKCs, nKCs])  # lateral inhibition connectivity matrix between KCs. Uniform for now
                                      # First dimension is the KC inhibited, second dimension is the KC inhibiting.
    inhscale[np.diag_indices(nKCs)] = 0  # Remove self-inhibition
    inhscale /= np.sum(inhscale, axis=1)[:, None]  # Normalize so that each KC receives the same amount of inhibition from all other KCs.
    inhscale_oe = inhscale.copy()
    inhscale *= inhfactor  # Ramp up the degree of inhibition just to showcase the effect.
    inhscale_oe *= inhfactor_oe  # Ramp up the degree of inhibition just to showcase the effect.

    # Stimulus
    ston = 3.6 # odor onset in seconds
    stdur = 5 # odor duration in seconds
    stoff = ston + stdur
    tend = stoff + 5
    tarr, starr = mdl.generate_step_stimulus(tend, ston, stoff, dt, 1 - bline) # Run a 5s long stimulus

    # Make the simulation serial for each chunk for memory sake, you need 2 values from each instance for each model type.
    resps = np.zeros([n_inps, nKCs, 4]) # steady state responses for each model type at odor offset

    for n in tqdm(range(int(n_inps / chunksize))):
        # Input scaling factor
        nrs = rng.poisson(prr*nKCs, chunksize) #random no of reliable responders, 2nd dimension for each different 'odor'
        nurs = rng.poisson(pur*nKCs, chunksize) #random no of unreliable responders, 2nd dimension for each different 'odor'
        inpscales = np.zeros([nKCs, chunksize])  # input scaling parameter for all 'different' odors.
        csprs = [[] for _ in range(chunksize)] # CS+ reliable responders
        cspus = [[] for _ in range(chunksize)] # CS+ reliable responders
        # Choose the reliable / unreliable responders for each instance
        for i in range(chunksize):
            csprs[i] = rng.choice(np.arange(nKCs), nrs[i], replace=False) # CS+ reliable responders
            cspus[i] = rng.choice(np.arange(nKCs)[list(set(np.arange(nKCs)) - set(csprs[i]))], nurs[i], replace=False) # CS+ unreliable responders
            inpscales[csprs[i], i] = rng.uniform(0.5, 1, nrs[i])
            inpscales[cspus[i], i] = rng.uniform(0, 0.5, nurs[i])  # unreliable responders


        # Activity arrays for all models
        CaKCcal = np.zeros([len(tarr), nKCs, chunksize])  # Calycal calcium transient array preallocated. Shape is time x KCs x CS+/-. This is also the no inhibition case!
        CaKClobe = np.zeros([len(tarr), nKCs, chunksize])  # MB lobe for only inhibition condition. Shape is time x KCs x CS+/-
        CaKClobe_nm = np.zeros([len(tarr), nKCs, chunksize])  # MB lobe for inhibition condition with activity-dependent modulation. Shape is time x KCs x CS+/-
        CaKClobe_oe = np.zeros([len(tarr), nKCs, chunksize])  # MB lobe for overexpression condition. Shape is time x KCs x CS+/-
        for ar in [CaKCcal, CaKClobe, CaKClobe_nm, CaKClobe_oe]:
            ar[0] = bline
        # Adaptation, inhibition.
        adapt = np.zeros([len(tarr), nKCs, chunksize])
        inh = np.zeros([len(tarr), nKCs, chunksize])
        inh_nm = np.zeros([len(tarr), nKCs, chunksize])
        inh_oe = np.zeros([len(tarr), nKCs, chunksize])

        # Noise (to test what happens to inhibition models with noise at the baseline)
        noise = rng.standard_normal([len(tarr)-1, nKCs, chunksize]) # Generate noise. Shape is time x KCs x CS+/-.

        # Simulate
        for i in range(len(tarr) - 1):
            # Calcium transients
            CaKCcal[i + 1] = CaKCcal[i] + ((inpscales * starr[i]) / tauinp - (CaKCcal[i] - bline) / tauKCdec) * dt
            CaKClobe[i + 1] = CaKClobe[i] + ((inpscales * starr[i]) / tauinp - (CaKClobe[i] - bline) / tauKCdec) * dt
            CaKClobe_nm[i + 1] = CaKClobe_nm[i] + ((inpscales * starr[i]) / tauinp - (CaKClobe_nm[i] - bline) / tauKCdec) * dt
            CaKClobe_oe[i + 1] = CaKClobe_oe[i] + ((inpscales * starr[i]) / tauinp - (CaKClobe_oe[i] - bline) / tauKCdec) * dt

            # ADD NOISE
            CaKCcal[i + 1] += noise[i] * n_str
            CaKClobe[i + 1] += noise[i] * n_str
            CaKClobe_nm[i + 1] += noise[i] * n_str
            CaKClobe_oe[i + 1] += noise[i] * n_str

            # Adaptation
            adapt[i + 1] = mdl.adaptation_dynamics(adapt[i], CaKCcal[i], tauadapt, adaptscale, dt)
            CaKCcal[i + 1] -= (adapt[i] * CaKCcal[i]) * (1 / tauKCdec) * dt
            CaKClobe[i + 1] -= (adapt[i] * CaKClobe[i]) * (1 / tauKCdec) * dt
            CaKClobe_nm[i + 1] -= (adapt[i] * CaKClobe_nm[i]) * (1 / tauKCdec) * dt
            CaKClobe_oe[i + 1] -= (adapt[i] * CaKClobe_oe[i]) * (1 / tauKCdec) * dt

            # inhibition (as in WT model)
            inh[i + 1] = mdl.inhibition_dynamics(inh[i], CaKClobe[i], tauinh, inhscale, dt) \
                         * mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe[i], infp, slf)  # modulated inhibition
            CaKClobe[i + 1] -= (inh[i]) * (1 / tauKCdec) * dt

            # inhibition (as in VI model)
            inh_nm[i + 1] = mdl.inhibition_dynamics(inh_nm[i], CaKClobe_nm[i], tauinh, inhscale, dt)  # nonmodulated inhibition
            CaKClobe_nm[i + 1] -= (inh_nm[i]) * (1 / tauKCdec) * dt

            # Overexpressed inhibition
            inh_oe[i + 1] = mdl.inhibition_dynamics(inh_oe[i], CaKClobe_oe[i], tauinh, inhscale_oe, dt) \
                            * mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_oe[i], infp, slf)  # modulated inhibition
            CaKClobe_oe[i + 1] -= (inh_oe[i]) * (1 / tauKCdec) * dt

            # Make sure everything is nonzero
            CaKCcal[i + 1] = np.clip(CaKCcal[i + 1], 0, None)
            CaKClobe[i + 1] = np.clip(CaKClobe[i + 1], 0, None)
            CaKClobe_nm[i + 1] = np.clip(CaKClobe_nm[i + 1], 0, None)
            CaKClobe_oe[i + 1] = np.clip(CaKClobe_oe[i + 1], 0, None)

        # Extract relevant stuff
        for i, ar in enumerate([CaKCcal, CaKClobe_nm, CaKClobe, CaKClobe_oe]):
            resps[chunksize*n:chunksize*(n+1), :, i] = ar[tarr == stoff].T.squeeze()

    # Save the baselines and responses so you do not rerun the whole stuff
    modelparams = {'nKCs': nKCs, 'tauKCdec': tauKCdec, 'tauinp': tauinp, 'tauadapt': tauadapt, 'tauinh': tauinh,
                   'adaptscale': adaptscale, 'inhscale': inhscale[0,1], 'inhscale_oe': inhscale_oe[0,1], 'infp': infp, 'slf': slf,
                   'bline': bline, 'dt': dt, 'prr': prr, 'pur': pur, 'n_str': n_str}
    sdir = r'../../data/model_related/odor_activity_dists'
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    np.savez(os.path.join(sdir, sname), modelparams=modelparams, resps=resps)


# Extract distributions
resps += rng.normal(nmean, nstd, resps.shape) # Add noise
resps = resps.reshape(n_inps, nKCs//nKC_per_avg, nKC_per_avg, 4).mean(axis=2)
# Make sure all is strictly positive by subtracting the minimum value
resps -= np.min(resps, axis=(0, 1))

resps_pooled = resps.reshape(-1, 4)
binsize = 0.01
bins = np.arange(0, resps_pooled.max(), binsize) # spacing now 0.01
hists = np.zeros([len(bins)-1, resps_pooled.shape[-1]]) # histogram for the second figure
for i in range(4):
    # Generate the histogram for each model type
    hist, _ = np.histogram(resps_pooled[:, i], bins=bins, density=False)
    hist = hist.astype(float)
    total_mass = np.nansum(hist)
    fraction_in_bins = total_mass / resps_pooled.shape[0]
    print(f"Fraction of total data in bins: {fraction_in_bins:.3f}")
    hist /= resps_pooled.shape[0]  # Normalize to the number of instances, now it is a percentage of the total data in each bin
    hists[:, i] = hist

# PLOTTING
# --------
fsd = r'../figures'

if not os.path.exists(fsd):
    os.makedirs(fsd)

# General plot parameters
colors = ['g', 'b', 'k', 'r']
labels = ['KD', 'VI', 'WT', 'OE']
# FIGURE 1: The activity distributions at odor odor offset for all model parameter
figdist, axdist = plt.subplots(1, 1, figsize=(12.59, 7.28))
for i in range(resps.shape[-1]):
    axdist.plot(bins[:-1]+binsize/2, hists[:,i], color=colors[i], lw=2, label=labels[i])

axdist.set_ylabel('Probability')
axdist.set_xlabel('Activity [a.u.]')
axdist.legend(loc=[0.94,0.68])
figdist.subplots_adjust(top=0.985, bottom=0.12, left=0.105, right=0.92, hspace=0.2, wspace=0.2)

plt.pause(0.1)
fn = 'figs4_odor_offset_activity_dist_overlaid'
[figdist.savefig(os.path.join(fsd, fn+'.%s'%extension)) for extension in extensions]
plt.close(figdist)
