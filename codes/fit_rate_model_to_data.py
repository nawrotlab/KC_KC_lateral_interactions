#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import KC_population_calcium_rate_model_functions as mdl
import fit_functions as ff
import plotting_related_functions
import pandas as pd
from scipy.optimize import curve_fit
from lmfit import minimize, Parameters, report_fit

rng = np.random.default_rng(666)

extensions = ['pdf', 'png'] # Figure extension for saving
tresp_cutoff = 11.2 # The cutoff time in s for plotting the reliable / unreliable responders.
tresp_on = 1.2 # The time in seconds when to start plotting.

# Fit the (new) rate model with independent rise and decay time constants to Manoim et al. 2022 data
# For the notes related to this script, see the (total mess) rise_decay_time_constants.py script.

# ONE CAVEAT WITH THE FITS: You do the fits WITHOUT NOISE and BASELINE!
# When you add noise, the baseline goes a little up but the stimulus-related parts stay mostly same values.
# Therefore, if you baseline normalize, the stimulus-related parts show a slight underrshoot in your case.

# You can think of trying the fits using baseline fixed at 0.2 (or whatever value you wish), but then you still need to subtract baseline
# from the model during objective function calculation. This is because the data is already normalized to have zero baseline.

# I tried the above point once and it did not produce nice fits for the WT model.


# Load data from Manoim 2022
dfr = pd.read_excel(r'../data/gamma_mch_responses_manoim_supplement.xlsx', header=[0,1]) # Multi-indexer
kddat = dfr['KD', 'Mean'].values # KD data
kderr = dfr['KD', 'SE'].values
wtdat = dfr['WT', 'Mean'].values # WT data
wterr = dfr['WT', 'SE'].values

# STIMULUS PARAMETERS
#--------------------
sf = 30 # sampling frequency in Herz
stondat = 3.6 # odor onset in seconds
stdatdur = 5 # odor duration in seconds
# time array
tdat = np.arange(0, len(dfr)/sf, 1/sf) # time array

# Fit to the decay after stimulus offset for the initial value of tauKCdec
fitfunc = lambda x, tau, a, c: a * np.exp(-x/tau) + c # Exponential decay function
# Fit for decay trace
fit_dec = [curve_fit(fitfunc, tdat[tdat >= stondat+stdatdur][np.argmax(ar):]-tdat[tdat >= stondat+stdatdur][np.argmax(ar)],
                     ar[np.argmax(ar):], p0=[1.5, ar[np.argmax(ar)], 0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))[0]
           for ar in [kddat[tdat >= stondat+stdatdur], wtdat[tdat >= stondat+stdatdur]]]

# GENERATE MODEL PARAMETER VARIABLES
#---------------------------
# Model hyperparameter(s)
normfac = 3.885 # L1 Normalization factor for the calcium traces

# Parameters shared by KD and WT models
params = Parameters()
params.add('tauKCdec', value=np.squeeze(fit_dec).mean(axis=0)[0], min=0)
params.add('tauinp', value=0.2, min=0)
params.add('tauadapt', value=2, min=0)
params.add('adaptscale', value=1, min=0)
params.add('bline', value=0, min=0, vary=False)

# Parameters specific to the WT model
inhparams = Parameters()
inhparams.add('tauinh', value=1.5, min=0)
inhparams.add('inhfactor', value=15, min=0)
inhparams.add('infp', value=0.5, min=1e-5)
inhparams.add('slf', value=0.03, min=1e-5)

# Stimulus related parameters
stimparams = Parameters()
stimparams.add('ststr', value=1, min=0, vary=False)
stimparams.add('ston', value=stondat, vary=False)
stimparams.add('stdur', value=stdatdur, min=0, vary=False)
stimparams.add('tend', value=tdat[-1], vary=False)
stimparams.add('dt', value=tdat[1]-tdat[0], vary=False)

# Fixed model parameters
n_units = 700  # Number of KCs
p_rr = 0.05    # Probability of a KC being a reliable responder
p_ur = 0.15    # Probability of a KC being an unreliable responder

# GENERATE THE MODEL VARIABLES
#----------------------------
# Reliable and unreliable responder parameters - Generate the input scaling array
rrs = rng.choice(np.arange(n_units), int(p_rr*n_units), replace=False)
urs = rng.choice(np.setdiff1d(np.arange(n_units), rrs), int(p_ur*n_units), replace=False)
inpscale = np.zeros(n_units)
inpscale[rrs] = rng.uniform(0.5, 1, len(rrs))
inpscale[urs] = rng.uniform(0, 0.5, len(urs))

# Generate odor stimulus
tarr, starr = mdl.generate_step_stimulus(stimparams['tend'].value, stimparams['ston'].value,
                                         stimparams['ston'].value+stimparams['stdur'].value,
                                         stimparams['dt'].value, stimparams['ststr'].value)

# DO THE FITTING
#---------------
# Fit the KD model
fit_ode_kd = minimize(ff.calculate_residual, params,
                   args=(tarr[tarr<=stimparams['ston'].value+stimparams['stdur'].value],
                         starr[tarr<=stimparams['ston'].value+stimparams['stdur'].value],
                         kddat[tarr<=stimparams['ston'].value+stimparams['stdur'].value],
                         kderr[tarr<=stimparams['ston'].value+stimparams['stdur'].value],
                         inpscale, None, stimparams['dt'].value, n_units, False, normfac),
                   method='lbfgsb') # NOW FIT IS 'TOO GOOD' with bullshit adaptation parameter values (like in the very beginning)
fittedpars_KD = fit_ode_kd.params.copy() # Fitted parameters

# Calculate noise as the residual between fit and data for the KD model
# Simulate the KD model population average trace
fittedvals_KD = ff.simulate_KD_model(tdat, starr, fittedpars_KD, inpscale, stimparams['dt'].value, n_units)[0] # Fitted values
res = np.mean(fittedvals_KD[tarr<=stimparams['ston'].value+stimparams['stdur'].value][:, inpscale > 0], axis=1) - \
       kddat[tarr<=stimparams['ston'].value+stimparams['stdur'].value]
n_str = np.std(res) # Noise strength i.e. the standard deviation of the OU process
fittedpars_KD.add('n_str', value=n_str, min=0) # Add noise strength to the parameters


# Fit the WT model
fit_ode_wt = minimize(ff.calculate_residual, inhparams,
                     args=(tarr[tarr<=stimparams['ston'].value+stimparams['stdur'].value],
                           starr[tarr<=stimparams['ston'].value+stimparams['stdur'].value],
                           wtdat[tarr<=stimparams['ston'].value+stimparams['stdur'].value],
                           wterr[tarr<=stimparams['ston'].value+stimparams['stdur'].value],
                           inpscale, fittedpars_KD, stimparams['dt'].value, n_units, True, normfac),
                     method='lbfgsb')
fittedpars_WT = fit_ode_wt.params.copy() # Fitted parameters

# Get the VI model fit by turning off the sigmoidal modulation.
# This can be practically done by setting the inflection point to a very large value (np.inf) and using the WT model simulation
fittedpars_VI = fittedpars_WT.copy() # For inhibition only
fittedpars_VI['infp'].value = np.inf # Turn off sigmoidal modulation

# Fit the OE model
params_OE = fittedpars_WT.copy() # Overexpression model
for p in params_OE:
    if p.startswith('inhfactor'):
        params_OE[p].value = fittedpars_WT[p].value # Initial guess

    else:
        params_OE[p].vary = False # Do not vary other parameters

# Simulate the voltage independent model population average trace to fit the OE model.
fittedvals_VI = ff.simulate_WT_model(tdat, starr, fittedpars_VI, fittedpars_KD, inpscale, stimparams['dt'].value, n_units)[0]
vidat = fittedvals_VI[:, inpscale>0].mean(axis=1)
# Fit the OE
fit_ode_oe = minimize(ff.calculate_residual, params_OE,
                     args=(tdat[tdat<=stimparams['ston'].value+stimparams['stdur'].value],
                           starr[tdat<=stimparams['ston'].value+stimparams['stdur'].value],
                           vidat[tdat<=stimparams['ston'].value+stimparams['stdur'].value],
                           1, # set the error to 1 since you are fitting to another model trace.
                           inpscale, fittedpars_KD, stimparams['dt'].value, n_units, True, 0),
                     method='lbfgsb')

fittedpars_OE = fit_ode_oe.params.copy() # Fitted parameters

# Report fit results
# KD model
print('KD model')
report_fit(fit_ode_kd)
# WT model
print('WT model')
report_fit(fit_ode_wt)
# OE model
print('OE model')
report_fit(fit_ode_oe)


# SAVE THE FIT RESULTS
#---------------------

# Create a parameter object for the rest of the parameters
restparams = Parameters()
restparams.add('nKCs', value=n_units, vary=False)
restparams.add('p_rr', value=p_rr, vary=False)
restparams.add('p_ur', value=p_ur, vary=False)
restparams.add('normfac', value=normfac, vary=False)

# Create the save directory
dsave = r'../data/model_related/fit_results' # Data save directory
if not os.path.exists(dsave):
    os.makedirs(dsave)

fname = 'model_fit_normfac_%.3f.npz'%(normfac)
# Save the fitted parameters
np.savez(os.path.join(dsave, fname),
         fittedpars_KD=fittedpars_KD, fittedpars_WT=fittedpars_WT, restparams=restparams, inhfactor_OE=fittedpars_OE['inhfactor'].value)


# Simulations with noise
#----------------------------------
n_str *= np.sqrt(2 / fittedpars_KD['tauKCdec'].value * stimparams['dt'].value) # Noise strength

# Generate a new stimulus with 2s burner time
stimparams.add('burner', value=2, vary=False)
tarr, starr = mdl.generate_step_stimulus(stimparams['tend'].value + stimparams['burner'].value, stimparams['ston'].value  + stimparams['burner'].value,
                                         stimparams['ston'].value+stimparams['stdur'].value  + stimparams['burner'].value,
                                         stimparams['dt'].value, stimparams['ststr'].value)
tarr -= stimparams['burner'].value  # remove burner time for figure plotting to remove initiation artifacts

# Generate noise
noise = rng.standard_normal([len(tarr)-1, n_units])  # Generate noise

# Simulate noisy models
# KD
fittedvals_KD_noisy = ff.simulate_KD_model(tarr, starr, fittedpars_KD, inpscale, stimparams['dt'].value, n_units, noise=noise, n_str=n_str)[0]
# WT
fittedvals_WT_noisy = ff.simulate_WT_model(tarr, starr, fittedpars_WT, fittedpars_KD, inpscale, stimparams['dt'].value, n_units, noise=noise, n_str=n_str)[0]
# VI
fittedvals_VI_noisy = ff.simulate_WT_model(tarr, starr, fittedpars_VI, fittedpars_KD, inpscale, stimparams['dt'].value, n_units, noise=noise, n_str=n_str)[0]
fittedvals_OE_noisy = ff.simulate_WT_model(tarr, starr, fittedpars_OE, fittedpars_KD, inpscale, stimparams['dt'].value, n_units, noise=noise, n_str=n_str)[0]

# Remove burner
fittedvals_KD_noisy = fittedvals_KD_noisy[tarr >= 0]
fittedvals_WT_noisy = fittedvals_WT_noisy[tarr >= 0]
fittedvals_VI_noisy = fittedvals_VI_noisy[tarr >= 0]
fittedvals_OE_noisy = fittedvals_OE_noisy[tarr >= 0]
starr = starr[tarr>=0]
tarr = tarr[tarr>=0]

# PLOTTING
#---------
fsd = r'../figures/' # Figure saving directory
if not os.path.exists(fsd):
    os.makedirs(fsd)


# Figure 1B: Plot the noisy overlaid traces for WT, VI and OE models
figfitover, axfitover = plt.subplots(figsize=(9.32, 4.8))
# Plot data & fits
for m, cfit, lab in zip([fittedvals_WT_noisy, fittedvals_VI_noisy, fittedvals_OE_noisy], ['k', 'b', 'r'], ['WT', 'VI', 'OE']):
    axfitover.plot(tdat, m[:, inpscale>0].mean(axis=1), cfit, lw=5, label=lab)

# Plot the stimulus
axfitover.plot(tdat[starr > 0], np.ones(len(tdat[starr > 0])) * -0.04, 'k-', lw=10)

# Figure adjustments
# Add labels
axfitover.set_xlabel('t [s]')
axfitover.set_ylabel('Activity [a.u.]')
# Add legend
axfitover.legend(loc='upper right', ncol=1)
# Adjust figure
figfitover.subplots_adjust(top=0.93, bottom=0.18, left=0.145, right=0.99)
# Save figure and close
plt.pause(0.01)
fn = 'fig1b_overlay_WT_VI_OE_models_noisy'
[figfitover.savefig(os.path.join(fsd, fn + '.%s' % figext)) for figext in extensions]
plt.close(figfitover)


# Figure S2F: Plot the sigmoidal used for modulating inhibition in the WT model
figsigm, axsigm = plt.subplots(figsize=(6.17, 4.3))
# Plot the sigmoidal
axsigm.plot(np.linspace(0,2.5, 1000),
            mdl.activity_dependent_inhibition_modulation_sigmoidal(np.linspace(0,2.5, 1000),
                                                                   fit_ode_wt.params['infp'].value,fit_ode_wt.params['slf'].value), 'k', lw=2)

# Figure adjustments
# Add labels
axsigm.set_xlabel('Activity [a.u.]')
axsigm.set_ylabel('Modulation factor')

# Adjust figure layout
figsigm.subplots_adjust(top=0.98, bottom=0.2, left=0.2, right=0.985, hspace=0.2, wspace=0.2)

# Save the outcome
plt.pause(0.01)
fn = 'figs2f_inhibition_modulation_sigmoidal'
[figsigm.savefig(os.path.join(fsd, fn + '.%s' % figext)) for figext in extensions]
plt.close(figsigm)


# Figure S2G: Population responses sorted by reliability
# TODO: THIS FILLS BETWEEN PERCENTILES, WHILE FOR THE LEARNING TRIALS I FILL BETWEEN STD! Choose one and stick to it!
#  other related script is rate_model_learning_mbon_normalized.py
figpop, axpop = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(16.9, 4.83))
tidx = (tdat >= tresp_on) & (tdat <= tresp_cutoff)# Time index for the reliable response cutoff
for i, tr in enumerate([fittedvals_KD_noisy, fittedvals_VI_noisy, fittedvals_WT_noisy, fittedvals_OE_noisy]):
    # Averages
    # Reliable responders
    axpop[i].plot(tdat[tidx]-tresp_on, tr[np.ix_(tidx, (inpscale>=0.5))].mean(axis=1), 'r', label='Reliable', lw=3)
    # Unreliable responders
    axpop[i].plot(tdat[tidx]-tresp_on, tr[np.ix_(tidx, ((inpscale<0.5) & (inpscale>0)))].mean(axis=1), 'orange', label='Unreliable', lw=3)
    # # Nonresponders
    # axpop[i].plot(tdat[tidx]-tresp_on, tr[np.ix_(tidx, (inpscale==0))].mean(axis=1), 'k', label='Non-responsive')

    # Shade the areas
    # Reliable responders
    axpop[i].fill_between(tdat[tidx]-tresp_on,
                          tr[np.ix_(tidx, (inpscale >= 0.5))].mean(axis=1) + tr[np.ix_(tidx, (inpscale >= 0.5))].std(axis=1),
                          np.clip(tr[np.ix_(tidx, (inpscale >= 0.5))].mean(axis=1) - tr[np.ix_(tidx, (inpscale >= 0.5))].std(axis=1), 0, None),
                          color='r', alpha=0.25)
    # Unreliable responders
    axpop[i].fill_between(tdat[tidx]-tresp_on,
                          tr[np.ix_(tidx, ((inpscale < 0.5) & (inpscale > 0)))].mean(axis=1) + tr[np.ix_(tidx, ((inpscale < 0.5) & (inpscale > 0)))].std(axis=1),
                          np.clip(tr[np.ix_(tidx, ((inpscale < 0.5) & (inpscale > 0)))].mean(axis=1) - tr[np.ix_(tidx, ((inpscale < 0.5) & (inpscale > 0)))].std(axis=1),
                          0,None), color='orange', alpha=0.25)

# Plot the stimulus
for ax in axpop:
    ax.plot(tdat[starr > 0]-tresp_on, np.ones(len(tdat[starr > 0])) * -0.04, 'k-', lw=5)
axpop[0].set_ylim([-0.05, 2.4])

# Add labels
axpop[1].set_xlabel('t [s]', x=1)
axpop[0].set_ylabel('Activity [a.u.]')

# Add titles (model variants)
for ax, lab in zip(axpop, ['KD', 'VI', 'WT', 'OE']):
    ax.set_title(lab)
# Add legend
axpop[1].legend(loc=[0.35,0.55])
# Adjust figure layout
figpop.subplots_adjust(top=0.88, bottom=0.185, left=0.06, right=0.99, wspace=0.09)

# Save figure and close
plt.pause(0.01)
fn = 'figs2g_model_fit_population_response_noisy'
[figpop.savefig(os.path.join(fsd, fn + '.%s' % figext)) for figext in extensions]
plt.close(figpop)
