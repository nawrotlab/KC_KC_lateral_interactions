#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import plotting_related_functions
import KC_population_calcium_rate_model_functions as mdl
import os
from sympy import nsolve
from sympy.abc import x,y,z

rng = np.random.default_rng(666) # Set random seed

# Get the MBON drive for the different model variants and save the results for further tuning of the MBON sigmoid transformation.
n_inps = 20 # Number of different input scaling cases for the model to average over

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

# Input scaling factor
nrs = rng.poisson(prr*nKCs, n_inps) #random no of reliable responders, 2nd dimension for each different 'odor'
nurs = rng.poisson(pur*nKCs, n_inps) #random no of unreliable responders, 2nd dimension for each different 'odor'
inpscales = np.zeros([nKCs, n_inps])  # input scaling parameter for all 'different' odors.
csprs = [[] for _ in range(n_inps)] # CS+ reliable responders
cspus = [[] for _ in range(n_inps)] # CS+ reliable responders
# Choose the reliable / unreliable responders for each instance
for i in range(n_inps):
    csprs[i] = rng.choice(np.arange(nKCs), nrs[i], replace=False) # CS+ reliable responders
    cspus[i] = rng.choice(np.arange(nKCs)[list(set(np.arange(nKCs)) - set(csprs[i]))], nurs[i], replace=False) # CS+ unreliable responders
    inpscales[csprs[i], i] = rng.uniform(0.5, 1, nrs[i])
    inpscales[cspus[i], i] = rng.uniform(0, 0.5, nurs[i])  # unreliable responders

# Stimulus
burner = 2
(tend, ston, stoff) = np.array([20, 5, 20]) + burner  # Time points for the test stimulus
tarr, starr = mdl.generate_step_stimulus(tend, ston, stoff, dt, 1 - bline) # Run a 5s long stimulus
# Correct for the burner time
tarr -= burner
tend -= burner
ston -= burner
stoff -= burner
# Activity arrays for all models
CaKCcal = np.zeros([len(tarr), nKCs, n_inps])  # Calycal calcium transient array preallocated. Shape is time x KCs x CS+/-. This is also the no inhibition case!
CaKClobe = np.zeros([len(tarr), nKCs, n_inps])  # MB lobe for only inhibition condition. Shape is time x KCs x CS+/-
CaKClobe_nm = np.zeros([len(tarr), nKCs, n_inps])  # MB lobe for inhibition condition with activity-dependent modulation. Shape is time x KCs x CS+/-
CaKClobe_oe = np.zeros([len(tarr), nKCs, n_inps])  # MB lobe for overexpression condition. Shape is time x KCs x CS+/-
for ar in [CaKCcal, CaKClobe, CaKClobe_nm, CaKClobe_oe]:
    ar[0] = bline

# Adaptation, inhibition.
adapt = np.zeros([len(tarr), nKCs, n_inps])
inh = np.zeros([len(tarr), nKCs, n_inps])
inh_nm = np.zeros([len(tarr), nKCs, n_inps])
inh_oe = np.zeros([len(tarr), nKCs, n_inps])

# Noise
noise = rng.standard_normal([len(tarr)-1, nKCs, n_inps]) # Generate noise. Shape is time x KCs x CS+/-.

# Generate the inhibition connectivity matrix
inhscale = np.ones([nKCs] * 2)  # lateral inhibition connectivity matrix between KCs. Uniform for now
inhscale[np.diag_indices(nKCs)] = 0  # Remove self-inhibition
inhscale /= np.sum(inhscale, axis=1)[:, None]  # Normalize so that each KC receives the same amount of inhibition from all other KCs.
inhscale_oe = inhscale.copy()
inhscale *= inhfactor  # Ramp up the degree of inhibition just to showcase the effect.
inhscale_oe *= inhfactor_oe  # Ramp up the degree of inhibition just to showcase the effect.

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

# Get the KC-MBON drive at the odor onset and offset to tune the MBONs
# For onset single value is sufficient since all model variants are the same ONLY IN THE NOISELESS CASE. I POSSIBLY NEED NOISE FOR INHIBITION MODELS.
(calon, caloff) = (np.sum(CaKCcal[tarr==ston].squeeze(), axis=0), np.sum(CaKCcal[tarr==stoff].squeeze(), axis=0))
(inon, inoff) = (np.sum(CaKClobe_nm[tarr==ston].squeeze(), axis=0), np.sum(CaKClobe_nm[tarr==stoff].squeeze(), axis=0))
(inmon, inmoff) = (np.sum(CaKClobe[tarr==ston].squeeze(), axis=0), np.sum(CaKClobe[tarr==stoff].squeeze(), axis=0))
(inoeon, inoeoff) = (np.sum(CaKClobe_oe[tarr==ston].squeeze(), axis=0), np.sum(CaKClobe_oe[tarr==stoff].squeeze(), axis=0))

# Save the results along with the model parameters
modelparams = {'nKCs': nKCs, 'tauKCdec': tauKCdec, 'tauinp': tauinp, 'tauadapt': tauadapt, 'tauinh': tauinh,
                'bline': bline, 'dt': dt, 'adaptscale': adaptscale, 'inhscale': inhscale[0,1], 'infp': infp, 'slf': slf, 'prr': prr, 'pur': pur, 'n_str': n_str}
sdir = r'../data/model_related/fit_results'
if not os.path.exists(sdir):
    os.makedirs(sdir)

np.savez(os.path.join(sdir, 'MBON_sigmoid_tuning_noisy.npz'), modelparams=modelparams, calon=calon, caloff=caloff,
                                                              inon=inon, inoff=inoff, inmon=inmon, inmoff=inmoff,
                                                              inoeon=inoeon, inoeoff=inoeoff)
