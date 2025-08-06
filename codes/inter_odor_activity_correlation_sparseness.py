#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import plotting_related_functions
import KC_population_calcium_rate_model_functions as mdl
import os
import pandas as pd
import scipy.stats as st
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import statsmodels.api as sm

rng = np.random.default_rng(666)  # Set random seed

# Simulate different model instances and calculate inter-odor activity correlation and sparseness of the KC population responses.

n_instances = 20 # Number of model instances
extensions = ['pdf', 'png'] # Figure save extension

# Load the fit parameters
fitparams = np.load(r'../data/model_related/fit_results/model_fit_normfac_3.885.npz')
tauKCdec, tauinp, tauadapt, adaptscale, bline, n_str = fitparams['fittedpars_KD']
tauinh, inhfactor, infp, slf = fitparams['fittedpars_WT']
inhfactor_oe = float(fitparams['inhfactor_OE'])
nKCs, p_rr, p_ur, __ = fitparams['restparams']
nKCs = int(nKCs)

# Remaining parameters not fitted to any data (mostly learning-related)
dt = 0.01  # time step
# Adjust noise strength to time constant
n_str = n_str * np.sqrt(2 / tauKCdec * dt)
inhscale = np.ones([nKCs, nKCs])  # lateral inhibition connectivity matrix between KCs.
                                  # First dimension is the KC inhibited, second dimension is the KC inhibiting.
inhscale[np.diag_indices(nKCs)] = 0  # Remove self-inhibition
inhscale /= np.sum(inhscale, axis=1)[:, None]  # Normalize so that each KC receives the same amount of inhibition from all other KCs.
inhscale_oe = inhscale.copy()
inhscale *= inhfactor
inhscale_oe *= inhfactor_oe

# Stimulus parameters
burner = 2
(tend, ston, stoff) = np.array([10, 5, 10]) + burner  # Time points for the test stimulus

# Stimulus generation
tarr, starr = mdl.generate_step_stimulus(tend, ston, stoff, dt, 1 - bline) # Run a 5s long stimulus
# Correct for the burner time
tarr -= burner
tend -= burner
ston -= burner
stoff -= burner


p_overlaps = ['random', 0.3, 0.4] # Overlap probabilities to test, random, 30% and 40% overlap.
fignames = ['s2b', 's2d', 's2c'] # Figure names for the overlap and sparseness
# Iterate over each overlap
for p_overlap, figname in zip(p_overlaps, fignames):
    # Responder probabilities
    p_response = p_rr + p_ur  # Probability of a KC responding to an odor
    if p_overlap == 'random':
        pass
    else:
        p_joint = p_response * p_overlap  # Joint probability of a KC responding to both odors
    p_rr_conditional = p_rr / p_response  # Conditional probability of a KC being a reliable responder given that it responds to an odor.
    p_ur_conditional = p_ur / p_response  # Conditional probability of a KC being an unreliable responder given that it responds to an odor

    responses_cal = np.zeros([n_instances, nKCs, 2]) # Responses for each instance, odor and KC at odor offset
    responses_lobe = np.zeros([n_instances, nKCs, 2]) # Responses for each instance, odor and KC at odor offset
    responses_lobe_im = np.zeros([n_instances, nKCs, 2]) # Responses for each instance, odor and KC at odor offset
    responses_lobe_oe = np.zeros([n_instances, nKCs, 2]) # Responses for each instance, odor and KC at odor offset for OE model
    overlap_percents = np.zeros([n_instances, 2]) # Overlap percentages for each instance and odor, shape n_instances x 2


    for q in range(n_instances):
        print(q)
        # YOU WILL NOW FIRST RUN THE ODOR RESPONSE SIMULATIONS, remove post odor simulations etc. for efficiency.
        # INITIALIZE ARRAYS
        # ------------------
        # Input scaling factor - NOTE THAT YOU USE inpscales defined later on, which is the concatenated version of both inpscale variables.
        inpscalecsp = np.zeros(nKCs)  # input scaling parameter for CS+, different for each KC.
        inpscalecsm = np.zeros(nKCs)  # input scaling parameter for CS-, different for each KC.


        # Adaptation
        adapt = np.zeros([len(tarr), nKCs, 2])  # adaptation array - shared by all since calyx-only.

        # Inhibition - not required since we will use the calyx simulations for correlations and sparseness business
        inh_nm = np.zeros([len(tarr), nKCs, 2])  # non-modulated lateral inhibition array
        inh_m = np.zeros([len(tarr), nKCs, 2])  # modulated lateral inhibition array
        inh_oe = np.zeros([len(tarr), nKCs, 2])

        # Noise
        noise = rng.standard_normal([len(tarr) - 1, nKCs, 2])  # Generate noise. Shape is time x KCs x CS+/-.

        # Calcium arrays - calyx should suffice here, since Bielopolski data is from calyx only!
        CaKCcal = np.zeros([len(tarr), nKCs, 2])  # Calycal calcium transient array preallocated. Shape is time x KCs x CS+/-. This is also the no inhibition case!
        CaKClobe_i = np.zeros([len(tarr), nKCs, 2])  # MB lobe for only inhibition condition. Shape is time x KCs x CS+/-
        CaKClobe_im = np.zeros([len(tarr), nKCs, 2])  # MB lobe for inhibition condition with activity-dependent modulation. Shape is time x KCs x CS+/-
        CaKClobe_oe = np.zeros([len(tarr), nKCs, 2])  # MB lobe for overexpression condition. Shape is time x KCs x CS+/-

        # Set the first values to baseline
        for ar in [CaKCcal, CaKClobe_i, CaKClobe_im, CaKClobe_oe]:
            ar[0] = bline

        # CHOOSE RANDOM RELIABLE AND UNRELIABLE RESPONDERS
        # --------------------------
        # Number of responders for CS+/-
        n_response = rng.poisson(p_response*nKCs, 2) #random no of responders, 1st val for CS+ and 2nd for CS-
        # Choose responders for the odors:
        # Choose first all indices for CS+, then choose the joint responders, finally choose the separate responders for CS- and concatenate with joint responders.
        r_csp = rng.choice(np.arange(nKCs), n_response[0], replace=False) # Responders for CS+
        if p_overlap == 'random':
            r_csm = rng.choice(np.arange(nKCs), n_response[1], replace=False) # Responders for CS- this case not caring about overlap
        else:
            n_joint = rng.poisson(p_joint * nKCs)  # random no of joint responders
            n_separate = np.clip(n_response - n_joint, 0,None)  # random no of separate responders (responding to one odor but not the other). First element CS+ other one CS-.
            r_joint = rng.choice(r_csp, n_joint, replace=False)  # Joint responders
            r_csm = np.concatenate([r_joint, rng.choice(np.arange(nKCs)[list(set(np.arange(nKCs)) - set(r_csp))], n_separate[1], replace=False)]) # Responders for CS-

        # Choose reliable / unreliable responders for CS+/- :
        # You have the conditional reliable / unreliable probabilities, so e.g for CS+ you can assign a uniform random value to each r_csp element and
        # take the ones that are smaller than or equal to p_rr_conditional as the resliable responders. Remaining ones are then unreliable responders.
        # This way you can implement both overlap between CS+/- and response reliability.
        # Note only you do not care about the specifics of the overlap, i.e. if reliable CS+ overlaps with unreliable CS- etc, since CS+/- overlap is independent of response reliability.
        # Therefore the expected number of KCs being reliable to both odors is nKCs * p_response * p_overlap * p_rr_conditional^2.
        csprs = r_csp[rng.random(n_response[0])<=p_rr_conditional]  # CS+ reliable responders
        cspus = np.array(list(set(r_csp) - set(csprs)))  # CS+ unreliable responders
        csmrs = r_csm[rng.random(n_response[1])<=p_rr_conditional]  # CS- reliable responders
        csmus = np.array(list(set(r_csm) - set(csmrs)))  # CS- unreliable responders

        print(' p_overlap = %s' % (p_overlap))
        print(' n_CS+ = %i, n_CS- = %i \n'
              ' n_CS+_rel = %i, n_CS+_unrel = %i \n'
              ' n_CS-_rel = %i, n_CS-_unrel = %i' %(n_response[0], n_response[1], len(csprs), len(cspus), len(csmrs), len(csmus)))
        print(' Response overlaps \n'
              ' TOTAL: CS+: %.2f%%, CS-: %.2f%%, n = %i \n'
              ' Reliable: CS+: %.2f%%, CS-: %.2f%%, n = %i \n'
              ' Unreliable: CS+: %.2f%%, CS-: %.2f%%, n = %i \n'
              ' Reliable vs unreliable: CS+: %.2f%%, CS-: %.2f%%, n = %i, %i \n'
              ' Unreliable vs reliable: CS+: %.2f%%, CS-: %.2f%%, n = %i, %i \n'
          %(len(set(r_csp).intersection(r_csm))/len(r_csp)*100, len(set(r_csm).intersection(r_csp))/len(r_csm)*100, len(set(r_csp).intersection(r_csm)),
            len(set(csprs).intersection(csmrs))/len(csprs)*100, len(set(csmrs).intersection(csprs))/len(csmrs)*100, len(set(csmrs).intersection(csprs)),
            len(set(cspus).intersection(csmus))/len(cspus)*100, len(set(csmus).intersection(cspus))/len(csmus)*100, len(set(csmus).intersection(cspus)),
            len(set(csprs).intersection(csmus))/len(csprs)*100, len(set(csmrs).intersection(cspus))/len(csmrs)*100, len(set(csprs).intersection(csmus)), len(set(csmrs).intersection(cspus)),
            len(set(cspus).intersection(csmrs))/len(cspus)*100, len(set(csmus).intersection(csprs))/len(csmus)*100, len(set(cspus).intersection(csmrs)), len(set(csmus).intersection(csprs))))

        overlap_percents[q, 0] = len(set(r_csp).intersection(r_csm)) / len(r_csp) * 100
        overlap_percents[q, 1] = len(set(r_csm).intersection(r_csp)) / len(r_csm) * 100

        # Update input scaling array
        inpscalecsp[csprs] = rng.uniform(0.5, 1, len(csprs))
        inpscalecsp[cspus] = rng.uniform(0, 0.5, len(cspus)) #unreliable responders
        inpscalecsm[csmrs] = rng.uniform(0.5, 1, len(csmrs))
        inpscalecsm[csmus] = rng.uniform(0, 0.5, len(csmus)) #unreliable responders

        inpscales = np.array([inpscalecsp, inpscalecsm]).T # Input scaling array for both CS+ and CS-. Shape nKCs x CS+/-

        # ODOR STIMULATION
        # -------------------
        for i in range(len(tarr) - 1):
            # Calcium transients
            CaKCcal[i + 1] = CaKCcal[i] + ((inpscales * starr[i]) / tauinp - (CaKCcal[i] - bline) / tauKCdec) * dt
            CaKClobe_im[i + 1] = CaKClobe_im[i] + ((inpscales * starr[i]) / tauinp - (CaKClobe_im[i] - bline) / tauKCdec) * dt
            CaKClobe_i[i + 1] = CaKClobe_i[i] + ((inpscales * starr[i]) / tauinp - (CaKClobe_i[i] - bline) / tauKCdec) * dt
            CaKClobe_oe[i + 1] = CaKClobe_oe[i] + ((inpscales * starr[i]) / tauinp - (CaKClobe_oe[i] - bline) / tauKCdec) * dt

            # ADD NOISE
            CaKCcal[i + 1] += noise[i] * n_str
            CaKClobe_im[i + 1] += noise[i] * n_str
            CaKClobe_i[i + 1] += noise[i] * n_str
            CaKClobe_oe[i + 1] += noise[i] * n_str

            # Adaptation
            adapt[i + 1] = mdl.adaptation_dynamics(adapt[i], CaKCcal[i], tauadapt, adaptscale, dt)
            CaKCcal[i + 1] -= (adapt[i] * CaKCcal[i]) * (1 / tauKCdec) * dt
            CaKClobe_im[i + 1] -= (adapt[i] * CaKClobe_im[i]) * (1 / tauKCdec) * dt
            CaKClobe_i[i + 1] -= (adapt[i] * CaKClobe_i[i]) * (1 / tauKCdec) * dt
            CaKClobe_oe[i + 1] -= (adapt[i] * CaKClobe_oe[i]) * (1 / tauKCdec) * dt

            # inhibition (as in WT model)
            inh_m[i + 1] = mdl.inhibition_dynamics(inh_m[i], CaKClobe_im[i], tauinh, inhscale, dt) \
                           * mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_im[i], infp, slf)  # modulated inhibition
            CaKClobe_im[i + 1] -= (inh_m[i]) * (1 / tauKCdec) * dt

            # inhibition (as in VI model)
            inh_nm[i + 1] = mdl.inhibition_dynamics(inh_nm[i], CaKClobe_i[i], tauinh, inhscale, dt)  # nonmodulated inhibition
            CaKClobe_i[i + 1] -= (inh_nm[i]) * (1 / tauKCdec) * dt

            # Overexpressed inhibition
            inh_oe[i + 1] = mdl.inhibition_dynamics(inh_oe[i], CaKClobe_oe[i], tauinh, inhscale_oe, dt) \
                            * mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_oe[i], infp, slf)  # modulated inhibition
            CaKClobe_oe[i + 1] -= (inh_oe[i]) * (1 / tauKCdec) * dt

            # Make sure everything is nonzero
            CaKCcal[i + 1] = np.clip(CaKCcal[i + 1], 0, None)
            CaKClobe_im[i + 1] = np.clip(CaKClobe_im[i + 1], 0, None)
            CaKClobe_i[i + 1] = np.clip(CaKClobe_i[i + 1], 0, None)
            CaKClobe_oe[i + 1] = np.clip(CaKClobe_oe[i + 1], 0, None)

        # Get the responses at odor offset
        responses_cal[q] = CaKCcal[tarr==stoff]
        responses_lobe[q] = CaKClobe_i[tarr==stoff]
        responses_lobe_im[q] = CaKClobe_im[tarr==stoff]
        responses_lobe_oe[q] = CaKClobe_oe[tarr==stoff]
        # print(st.pearsonr(CaKCcal[(tarr>=ston)&(tarr<=stoff),:,0].flatten(), CaKCcal[(tarr>=ston)&(tarr<=stoff),:,1].flatten()))

    print('Degree of overlap : %s' % p_overlap)
    # Check if overlaps are significantly different between CS+ and CS- odors
    ttest = st.ttest_rel(*overlap_percents.T)
    print('Overlap CS+ vs CS-: %s, p = %.2f' %(ttest.statistic, ttest.pvalue))
    if ttest.pvalue < 0.05:
        raise Warning('Overlap between CS+ and CS- is significantly different!')

    # Calculate the correlations and sparseness
    corrs = np.zeros([n_instances, 4]) # Correlations between CS+ and CS- responses
    sparseness = np.zeros([n_instances, 4, 2]) # Sparseness of the responses. Shape n instances, n model type and CS +/-
    for i, resp in enumerate([responses_cal, responses_lobe, responses_lobe_im, responses_lobe_oe]):
        corrs[:, i] = np.array([st.pearsonr(resp[j, :, 0], resp[j, :, 1])[0] for j in range(n_instances)]) # Correlations between CS+ and CS- responses
        sparseness[:, i] = 1 / (1 - 1/nKCs) * (1 - (np.sum(resp/nKCs, axis=1)**2 / np.sum(resp**2/nKCs, axis=1)))

    # Create dataframes for the subsequent testing
    labels = ['KD', 'VI', 'WT', 'OE']
    dfrcorr = pd.DataFrame([{"Instance": i, "Model": labels[j], "Correlation": corrs[i, j]}
                        for i in range(corrs.shape[0])
                        for j in range(corrs.shape[1])])
    dfrspars = pd.DataFrame([{"Instance": i, "Model": labels[j], "Odor": ['CS+', 'CS-'][k], "Sparseness": sparseness[i, j, k]}
                        for i in range(sparseness.shape[0])
                        for j in range(sparseness.shape[1])
                        for k in range(sparseness.shape[2])])

    dfrpover = pd.DataFrame([{"Instance": i, "Odor": ['CS+', 'CS-'][k], "Overlap": overlap_percents[i, k]}
                        for i in range(overlap_percents.shape[0])
                        for k in range(overlap_percents.shape[1])])

    # Correlation
    print('CORRELATION')
    model = ols('Correlation ~ C(Model)', data=dfrcorr).fit()
    print(sm.stats.anova_lm(model))
    test = pairwise_tukeyhsd(dfrcorr['Correlation'], dfrcorr['Model'], alpha=0.01)
    csppvals = test.pvalues
    print(test.summary())

    # Sparseness
    print('SPARSENESS')
    print('CS+')
    model = ols('Sparseness ~ C(Model)', data=dfrspars[dfrspars.Odor=='CS+']).fit()
    print(sm.stats.anova_lm(model))
    test = pairwise_tukeyhsd(dfrspars[dfrspars.Odor=='CS+']['Sparseness'], dfrspars[dfrspars.Odor=='CS+']['Model'], alpha=0.01)
    print(test.summary())
    print('CS-')
    model = ols('Sparseness ~ C(Model)', data=dfrspars[dfrspars.Odor=='CS-']).fit()
    print(sm.stats.anova_lm(model))
    test = pairwise_tukeyhsd(dfrspars[dfrspars.Odor=='CS-']['Sparseness'], dfrspars[dfrspars.Odor=='CS-']['Model'], alpha=0.01)
    print(test.summary())
    # Test for difference between CS+ and CS- sparseness
    print('CS+ vs CS-')
    print(st.ttest_ind(dfrspars[dfrspars.Odor=='CS+']['Sparseness'], dfrspars[dfrspars.Odor=='CS-']['Sparseness']))
    print('WT vs VI odor pooled')
    print(st.ttest_rel(dfrspars[dfrspars.Model=='WT']['Sparseness'], dfrspars[dfrspars.Model=='VI']['Sparseness']))

    # Bring ALL the dataframes together to save them
    sdir = r'../data/model_related/overlap_correlation_sparseness'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    # Merge correlation and sparseness dataframes
    dfsave = pd.merge(dfrcorr, dfrspars, on=['Instance', 'Model'])
    # Merge with overlap dataframe
    dfsave = pd.merge(dfsave, dfrpover, on=['Instance', 'Odor'])
    # Save the dataframe
    dfsave.to_csv(os.path.join(sdir, 'correlation_sparseness_overlap_%s.csv' %p_overlap), index=False)

    # PLOTTING
    # --------
    # Figure save directory
    fsd = r'../figures'
    if not os.path.exists(fsd):
        os.makedirs(fsd)

    colors = ['g', 'b', 'k', 'r']

    # Figure 1: Inter-odor activity correlation
    figcor, axcor = plt.subplots(figsize=(3.9,7))

    # Plot the correlations
    for i in range(4):
        # All data points
        axcor.scatter(np.ones(n_instances)*(i+1)/2 + rng.normal(0, 0.025, n_instances), corrs[:, i],  color=colors[i], label=labels[i])
        # Average as horizontal line
        axcor.plot([(i+1)/2-0.1, (i+1)/2+0.1], [np.mean(corrs[:, i]), np.mean(corrs[:, i])], color=colors[i], lw=2)

    # Plot adjustments
    # Adjust x axis
    axcor.set_xticks([0.5, 1, 1.5, 2])
    axcor.set_xticklabels(labels)
    # axcor.set_xlim(0.25,1.75)
    # Adjust y ticks
    axcor.set_yticks([0,0.1,0.2])
    # Label y axis
    axcor.set_ylabel('Correlation')

    # Adjust figure
    figcor.subplots_adjust(top=0.955, bottom=0.08, left=0.315, right=0.97)

    plt.pause(0.01)
    [figcor.savefig(os.path.join(fsd, 'fig%s_inter_odor_activity_correlation.'%figname + extension)) for extension in extensions]
    plt.close(figcor)

    # Figure 2: Sparseness of the responses for each model type
    figspar, axspar = plt.subplots(figsize=(5.6,7))

    # Plot the sparseness
    for i in range(4):
        for j in range(2):
            # All data points
            axspar.scatter(np.ones(n_instances) * (j + 1) * 0.6 + i / 10 + rng.normal(0, 0.012, n_instances), sparseness[:, i, j], color=colors[i], label=labels[i])
            # Average as horizontal line
            axspar.plot([(j + 1) * 0.6 + i / 10 - 0.05, (j + 1) * 0.6 + i / 10 + 0.05], [np.mean(sparseness[:, i, j]), np.mean(sparseness[:, i, j])], color=colors[i], lw=2)


    # Plot adjustments
    # Adjust x axis
    axspar.set_xticks([0.75, 1.35])
    axspar.set_xticklabels([r'$CS^+$', r'$CS^-$'])
    # axspar.set_xlim(0.4, 1.3)
    # Adjust y axis
    axspar.set_ylim(0.79, 0.97)
    axspar.set_yticks(np.linspace(0.8,0.95,4))
    axspar.set_ylabel('Sparseness')

    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(4)]
    axspar.legend(handles, labels, loc=[0.8,0.05])

    # Adjust figure size
    figspar.subplots_adjust(top=0.955, bottom=0.08, left=0.245, right=0.84)

    plt.pause(0.01)
    [figspar.savefig(os.path.join(fsd, 'fig%s_sparseness.'%figname + extension)) for extension in extensions]
    plt.close(figspar)