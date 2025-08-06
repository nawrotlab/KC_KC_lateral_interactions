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

extensions = ['png', 'pdf'] # Figure save extension

learnrates = np.linspace(0.06, 0.26, 11) # learning rate specifying the synaptic weight change magnitude.
# Readjust the first and last entries
learnrates[0] = 0.05
learnrates[-1] = 0.25
n_instances = 20 # Number of model instances to run for each learning rate

# Load the fit parameters
fitparams = np.load(r'../data/model_related/fit_results/model_fit_normfac_3.885.npz')
tauKCdec, tauinp, tauadapt, adaptscale, bline, n_str = fitparams['fittedpars_KD']
tauinh, inhfactor, infp, slf = fitparams['fittedpars_WT']
inhfactor_oe = float(fitparams['inhfactor_OE'])
nKCs, p_rr, p_ur, __ = fitparams['restparams']
nKCs = int(nKCs)

# Remaining parameters not fitted to any data (mostly learning-related)
taudan = 2  # dopaminergic neuron time constant in s -> making this longer makes pragmatically sense because it leads to
              # a larger learning window. It is also biologically much more plausible.
dt = 0.01  # time step this should be sufficient actually
# Adjust noise strength to time constant
n_str = n_str * np.sqrt(2 / tauKCdec * dt)
# Generate the inhibition connectivity matrix
inhscale = np.ones([nKCs] * 2)  # lateral inhibition connectivity matrix between KCs. Uniform for now
inhscale[np.diag_indices(nKCs)] = 0  # Remove self-inhibition
inhscale /= np.sum(inhscale, axis=1)[:, None]  # Normalize so that each KC receives the same amount of inhibition from all other KCs.
inhscale_oe = inhscale.copy()
inhscale *= inhfactor  # Ramp up the degree of inhibition just to showcase the effect.
inhscale_oe *= inhfactor_oe  # Ramp up the degree of inhibition just to showcase the effect.

# Stimulus and shock parameters
burner = 2        # burner time in s
ston = 5 + burner # add burner time for figure plotting to remove initiation artifacts
stoff = ston + 60
tend = stoff + 10
nshocks = 12 # Number of shocks
shdur = 1.25 # Shock duration
shint = 3.75 # Inter-shock interval
shstr = 0.5  # shock strength

# Stimulus, shock
tarr, starr = mdl.generate_step_stimulus(tend, ston, stoff, dt, 1 - bline)  # Odor stimulus
__, sharr = mdl.generate_stimulus_stream(ston + shint / 2, nshocks, shdur, shint, shstr, 10 + shint / 2+dt, dt)  # Shock stimulus
tarr -= burner  # Remove burner time from the time array
ston -= burner  # Remove burner time from the stimulus onset
stoff -= burner  # Remove burner time from the stimulus offset
tend -= burner  # Remove burner time from the end time

# Learning etc parameters
maxweight = 1    # maximum KC::MBON synaptic weight initially

# MBON activity normalization parameters
sup = 1.5 # The supremum of the sigmoidal function i.e. the amplitude from zero crossing (baseline MBON drive)
mbub = 1  # The MBON activity upper bound for the solver

# Get the MBON sigmoidal normalization parameters
# Load the MBON activity normalization parameters - RUN FIRST rate_model_get_summed_KC_activity_different_model_variants to save MBON tuning results.
loader = np.load(r'../data/model_related/fit_results/MBON_sigmoid_tuning_noisy.npz', allow_pickle=True)

# Maximum MBON activities when linear sum of KC activities for a given odor, this considers the average KC-MBON drive at odor offset over 20 different initializations.
maxmbs = np.array([loader['caloff'].mean(), loader['inoff'].mean(), loader['inmoff'].mean(), loader['inoeoff'].mean()])
# Add an offset to account for the baseline MBON activities not being 0 -> inflection point is between max and min activities.
minmbs = np.array([loader['calon'].mean(), loader['inon'].mean(), loader['inmon'].mean(), loader['inoeon'].mean()])

# Solve numerically for the maximum weight that would lead to a 0.9 MBON activity for the untrained odor, 0.5 in the middle and ~0 in non stimulus baseline.
sols = nsolve([x + (sup-x)/(1+np.e**(y/z)), x - 0.5 + (sup-x)/(1+np.e**((y-0.5)/z)), x - mbub + (sup-x)/(1+np.e**((y-1)/z))], [x,y,z],
               [0,0.5,0.09])
mnmbon, infpmbon, slfmbon = [float(s) for s in sols]


# TODO: You will loop over the overlaps of 0.3 and 0.4, generating figures and saving everything on the go.
p_overlaps = [0.3, 0.4] # Overlap probabilities to test, random, 30% and 40% overlap.
for p_overlap in p_overlaps:
    valences = np.zeros([n_instances, len(learnrates), 4, 2]) # Valences to be saved for all cases, shape n_instances x n_learnrates x n_modelvariants x n_CS+/CS-.

    # Responder probabilities
    p_response = p_rr + p_ur # Probability of a KC responding to an odor
    p_joint = p_response * p_overlap # Joint probability of a KC responding to both odors
    p_rr_conditional = p_rr / p_response # Conditional probability of a KC being a reliable responder given that it responds to an odor.
    p_ur_conditional = p_ur / p_response # Conditional probability of a KC being an unreliable responder given that it responds to an odor

    print('p overlap: %.1f' %p_overlap)
    for q in range(n_instances):
        print(q)
        # INITIALIZE ARRAYS
        # ------------------
        inpscalecsp = np.zeros(nKCs)  # input scaling parameter for CS+, different for each KC.
        inpscalecsm = np.zeros(nKCs)  # input scaling parameter for CS-, different for each KC.

        # Adaptation
        adapt = np.zeros([len(tarr), nKCs, 2])  # adaptation array - shared by all since calyx-only.

        # Inhibition
        inh_nm = np.zeros([len(tarr), nKCs, 2])  # non-modulated lateral inhibition array
        inh_m = np.zeros([len(tarr), nKCs, 2])  # modulated lateral inhibition array
        inh_oe = np.zeros([len(tarr), nKCs, 2])

        # Noise
        noise = rng.standard_normal([len(tarr) - 1, nKCs, 2])  # Generate noise. Shape is time x KCs x CS+/-.

        # Calcium arrays
        CaKCcal = np.zeros([len(tarr), nKCs, 2])  # Calycal calcium transient array preallocated. Shape is time x KCs x CS+/-. This is also the no inhibition case!
        CaKClobe_i = np.zeros([len(tarr), nKCs, 2])  # MB lobe for only inhibition condition. Shape is time x KCs x CS+/-
        CaKClobe_im = np.zeros([len(tarr), nKCs, 2])  # MB lobe for inhibition condition with activity-dependent modulation. Shape is time x KCs x CS+/-
        CaKClobe_oe = np.zeros([len(tarr), nKCs, 2])  # MB lobe for overexpression condition. Shape is time x KCs x CS+/-

        # Set the first values to baseline
        for ar in [CaKCcal, CaKClobe_i, CaKClobe_im, CaKClobe_oe]:
            ar[0] = bline

        # CHOOSE RANDOM RELIABLE AND UNRELIABLE RESPONDERS
        # --------------------------
        # Number of responders
        n_response = rng.poisson(p_response*nKCs, 2) #random no of responders, 1st val for CS+ and 2nd for CS-
        n_joint = rng.poisson(p_joint*nKCs) #random no of joint responders
        n_separate = np.clip(n_response - n_joint, 0, None) #random no of separate responders (responding to one odor but not the other). First element CS+ other one CS-.

        # Choose responders for the odors:
        # Choose first all indices for CS+, then choose the joint responders, finally choose the separate responders for CS- and concatenate with joint responders.
        r_csp = rng.choice(np.arange(nKCs), n_response[0], replace=False) # Responders for CS+
        r_joint = rng.choice(r_csp, n_joint, replace=False) # Joint responders
        r_csm = np.concatenate([r_joint, rng.choice(np.arange(nKCs)[list(set(np.arange(nKCs)) - set(r_csp))], n_separate[1], replace=False)]) # Responders for CS-

        # Choose reliable / unreliable responders for CS+/- :
        csprs = r_csp[rng.random(n_response[0])<=p_rr_conditional]  # CS+ reliable responders
        cspus = np.array(list(set(r_csp) - set(csprs)))  # CS+ unreliable responders
        csmrs = r_csm[rng.random(n_response[1])<=p_rr_conditional]  # CS- reliable responders
        csmus = np.array(list(set(r_csm) - set(csmrs)))  # CS- unreliable responders

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

        # Update input scaling array
        inpscalecsp[csprs] = rng.uniform(0.5, 1, len(csprs))
        inpscalecsp[cspus] = rng.uniform(0, 0.5, len(cspus)) #unreliable responders
        inpscalecsm[csmrs] = rng.uniform(0.5, 1, len(csmrs))
        inpscalecsm[csmus] = rng.uniform(0, 0.5, len(csmus)) #unreliable responders

        inpscales = np.array([inpscalecsp, inpscalecsm]).T # Input scaling array for both CS+ and CS-. Shape nKCs x CS+/-

        # LEARNING SIMULATION
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

        # LEARNING SIMULATIONS for DIFFERENT LEARNING RATES
        for j, learnrate in enumerate(learnrates):
            print('Learning rate: %.2f' %learnrate)
            # RUN THE LEARNING SIMULATIONS for the SAME SET OF MODEL ACTIVITIES
            # DAN
            dan = np.zeros([len(tarr), 4])  # DAN activity array.

            # KC::MBON synaptic weight arrays - only for CS+ since CS- undergoes no change.
            synw_ni = np.zeros([len(tarr), nKCs])  # weight change no inhibition
            synw_i = np.zeros([len(tarr), nKCs])  # weight change only inhibition
            synw_im = np.zeros([len(tarr), nKCs])  # weight change inhibition and activity-dependent modulation
            synw_oe = np.zeros([len(tarr), nKCs]) # weight change WT overexpression model
            for wght in [synw_ni, synw_i, synw_im, synw_oe]:
                wght[0, :] = maxweight  # Set the initial weights to maxweight for all conditions

            # MBON activity and shock valence prediction arrays
            mbonacts_csp = np.zeros([len(tarr), 4])  # Time-resolved MBON activity for all model variants for the CS+ stimulation.
            mbonacts_csm = np.zeros([len(tarr), 4])  # Time-resolved MBON activity for all model variants for the CS- stimulation.
            valpreds = np.zeros([len(tarr), 4])
            valpredscsm = np.zeros([len(tarr), 4])

            for i in range(len(tarr) - 1):
                # Learning related processes: MBON activities & DAN dynamics for all model variants -> loop over each model variant
                for midx, (cval, wval) in enumerate(zip([CaKCcal, CaKClobe_i, CaKClobe_im, CaKClobe_oe],
                                                        [synw_ni[i, :], synw_i[i, :], synw_im[i, :], synw_oe[i, :]])):
                    #MBON activities - make sure nonzero
                    mbonacts_csp[i, midx] = np.clip(mdl.sigmoidal_func(((cval[i, :, 0] @ wval) - minmbs[midx])/(maxmbs[midx]-minmbs[midx]), slfmbon, infpmbon, mnmbon, sup-mnmbon),
                                                    0, None) # CS+ MBON activity
                    mbonacts_csm[i, midx] = np.clip(mdl.sigmoidal_func(((cval[i, :, 1] @ wval) - minmbs[midx])/(maxmbs[midx]-minmbs[midx]), slfmbon, infpmbon, mnmbon, sup-mnmbon),
                                                    0, None) # CS+ MBON activity
                    # Untrained MBON activity with sigmoid transform - this corresponds (for now) to an appetitive compartment MBON where no learning occurs.
                    mb_ut = np.clip(mdl.sigmoidal_func(((np.sum(cval[i, :, 0]) * maxweight) - minmbs[midx]) / (maxmbs[midx] - minmbs[midx]), slfmbon, infpmbon, mnmbon, sup - mnmbon),
                                    0, None)
                    # Same as above but for CS- odor.
                    mb_ut_csm = np.clip(mdl.sigmoidal_func(((np.sum(cval[i, :, 1]) * maxweight) - minmbs[midx]) / (maxmbs[midx] - minmbs[midx]), slfmbon, infpmbon, mnmbon, sup - mnmbon),
                                    0, None)

                    # VALENCE PREDICTION
                    # Predicted valence is the activity difference between trained and untrained MBONs.
                    valpred = mb_ut - mbonacts_csp[i, midx] #ANNA'S PAPER CAN JUSTIFY CALCULATING VALENCE AS THIS
                    # Relative between untrained and current (CS-) MBON activity.
                    valpredcsm = mb_ut_csm - mbonacts_csm[i, midx]

                    # Update predicted valences
                    valpreds[i, midx] = valpred
                    valpredscsm[i, midx] = valpredcsm

                    # DAN dynamics
                    dan[i + 1, midx] = mdl.DAN_dynamics(dan[i, midx], taudan, dt, sharr[i], valpred)

                # Update the synaptic weights
                synw_ni[i+1] = mdl.KC_MBON_coincidence_based_weight_change(synw_ni[i], CaKCcal[i, :, 0], dan[i, 0], dt, learnrate=learnrate)
                synw_i[i+1] = mdl.KC_MBON_coincidence_based_weight_change(synw_i[i], CaKClobe_i[i, :, 0], dan[i, 1], dt, learnrate=learnrate)
                synw_im[i+1] = mdl.KC_MBON_coincidence_based_weight_change(synw_im[i], CaKClobe_im[i, :, 0], dan[i, 2], dt, learnrate=learnrate)
                synw_oe[i + 1] = mdl.KC_MBON_coincidence_based_weight_change(synw_oe[i], CaKClobe_oe[i, :, 0], dan[i, 3], dt, learnrate=learnrate)

            # Correct the MBON activity in the last time instance
            for idx, (tr, wg) in enumerate(zip([CaKCcal[-1], CaKClobe_i[-1], CaKClobe_im[-1], CaKClobe_oe[-1]],
                                               [synw_ni[-1, :], synw_i[-1, :], synw_im[-1, :], synw_oe[-1, :]])):
                mbonacts_csp[-1, idx] = np.clip(mdl.sigmoidal_func((tr[:, 0] @ wg - minmbs[idx]) / (maxmbs[idx] - minmbs[idx]), slfmbon, infpmbon, mnmbon,
                                       sup - mnmbon), 0, None)  # CS+ MBON activity
                mbonacts_csm[-1, idx] = np.clip(mdl.sigmoidal_func((tr[:, 1] @ wg - minmbs[idx]) / (maxmbs[idx] - minmbs[idx]), slfmbon, infpmbon, mnmbon,
                                       sup - mnmbon), 0, None)  # CS- MBON activity

            valences[q, j] = np.vstack([valpreds[tarr==stoff], valpredscsm[tarr==stoff]]).T

    savedir = r'../data/model_related/learning_rate_simulations'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    filename = 'valences_learning_rate_screening_p_overlap_%.1f_shock_strength_%.2f_inhfactor_%i.npz' %(p_overlap, shstr, inhfactor)
    np.savez(os.path.join(savedir, filename), valences=valences, learnrates=learnrates)

    if p_overlap > 0.3:
        continue # Figure S2H shows only the 0.3 overlap case

    # PLOTTING
    # --------
    # Figure save directory etc
    fsd = r'../figures'
    if not os.path.exists(fsd):
        os.makedirs(fsd)


    colors = ['g', 'b', 'gray', 'red']
    labels = ['KD', 'VI', 'WT', 'OE']

    # Figure 1: CS+ final valences for different learning rates and model variants
    # Create a figure and axis
    figvls, axvls = plt.subplots(2,1, sharex=True, figsize=(17, 8))
    figvls.canvas.manager.full_screen_toggle()

    # Number of groups (x-axis) and cases
    num_groups = valences.shape[1]  # 10 groups
    num_cases = valences.shape[2]  # 4 cases
    num_distributions = valences.shape[0]  # 5 values per group

    fn = 'figs2h_prediction_error_boxplots.'

    # Offset for spacing between cases
    width = 0.2  # Width of each case's boxplot
    positions = np.arange(num_groups)  # Base positions for the 10 groups on the x-axis

    # Loop through the cases and create boxplots
    bxs = []
    for case in range(num_cases):
        # Create the boxplots for CS+
        bp1 = axvls[0].boxplot(
            [shstr - valences[:, i, case, 0] for i in range(num_groups)],  # Extract distributions for each group
            positions=positions + case * width - width * (num_cases - 1) / 2,  # Offset for spacing
            widths=width,  # Width of the boxplots
            patch_artist=True,  # Enable face color
            medianprops=dict(color="black"),  # Customize median line
            showfliers=True,  # Optionally hide outliers
          )
        # Change the boxplot colors
        for patch in bp1['boxes']:
            patch.set_facecolor(colors[case])
        #
        bp2 = axvls[1].boxplot(
            [0 - valences[:, i, case, 1] for i in range(num_groups)],  # Extract distributions for each group
            positions=positions + case * width - width * (num_cases - 1) / 2,  # Offset for spacing
            widths=width,  # Width of the boxplots
            patch_artist=True,  # Enable face color
            medianprops=dict(color="black"),  # Customize median line
            showfliers=True,  # Optionally hide outliers
          )
        # Change the boxplot colors https://stackoverflow.com/questions/41997493/python-matplotlib-boxplot-color
        for patch in bp2['boxes']:
            patch.set_facecolor(colors[case])

        bxs.append(bp2['boxes'][0])


    # Axis adjustments
    axvls[-1].set_xticks(positions)
    axvls[-1].set_xticklabels(learnrates.round(2))
    axvls[-1].set_xlabel('Learning rate')
    axvls[0].set_ylabel('Final prediction error')
    axvls[0].yaxis.set_label_coords(-0.075, 0)
    # Add titles
    for a, tit in zip(axvls, [r'$CS^+$', r'$CS^-$']):
        a.set_title(tit)
    # Add a legend
    axvls[0].legend(bxs, labels, loc='upper right')

    [figvls.savefig(os.path.join(fsd, fn+ext)) for ext in extensions]
    plt.close(figvls)