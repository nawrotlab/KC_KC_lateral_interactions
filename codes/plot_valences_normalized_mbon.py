#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, r'../function_scripts')
import plotting_related_functions
import os
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import statsmodels.api as sm


# Plot the valences for the learning rate screening simulations - RUN AFTER learning_rate_screening_mbon_normalized.py

# General parameters
extensions = ['png', 'pdf'] # The figure extension
shstr = 0.5 # The shock strength
inhfactor = 15 # The inhibition factor
learnrate = 0.12 # Learning rate
ldir = r'../data/model_related/learning_rate_simulations' # data load directory
p_overlaps = [0.3, 0.4] # Expected degree of overlap
fignames = ['1c', 's2e']

for p_overlap, figname in zip(p_overlaps, fignames):
    fname = 'valences_learning_rate_screening_p_overlap_%.1f_shock_strength_%.2f_inhfactor_%i.npz'%(p_overlap,shstr,inhfactor)
    loader = np.load(os.path.join(ldir,fname))
    valences = loader['valences'] # Shape n instances x n learning rates x n models x CS+/-
    learning_rates = loader['learnrates']

    valence = valences[:, np.argwhere(learning_rates==learnrate)].squeeze() # The valence we are interested in for the analysis and barplot

    # Create first a dataframe with the valences for the further anova
    instances = np.arange(valence.shape[0]) # The model instances
    models = ['naive', 'inh', 'mod_inh', 'over_exp'] # The model types
    odors = ['CS+', 'CS-'] # The odors

    # Create a flat DataFrame
    dfr = pd.DataFrame([{"Instance": instances[i], "Model": models[j], "Odor": odors[k], "Valence": valence[i, j, k]}
                        for i in range(valence.shape[0])
                        for j in range(valence.shape[1])
                        for k in range(valence.shape[2])])

    print('p overlap: %.1f'%p_overlap)

    # STATISTICAL ANALYSIS
    #---------------------
    print('CS+')
    # 1 WAY ANOVA
    model = ols('Valence ~ C(Model)', data=dfr[dfr.Odor=='CS+']).fit()
    print(sm.stats.anova_lm(model))
    # Do the Tukey's HSD test
    test = pairwise_tukeyhsd(dfr[dfr.Odor=='CS+']['Valence'], dfr[dfr.Odor=='CS+']['Model'], alpha=0.01)
    csppvals = test.pvalues
    print(test.summary())
    # CS-
    print('CS-')
    # 1 WAY ANOVA
    model = ols('Valence ~ C(Model)', data=dfr[dfr.Odor=='CS-']).fit()
    print(sm.stats.anova_lm(model))

    # Do the Tukey's HSD test
    test = pairwise_tukeyhsd(dfr[dfr.Odor=='CS-']['Valence'], dfr[dfr.Odor=='CS-']['Model'], alpha=0.01)
    csmpvals = test.pvalues
    print(test.summary())

    # PLOTTING
    #---------
    labels = ['KD', 'VI', 'WT', 'OE']
    # General colors for no inh, inh and inh_modulated
    colors = ['g', 'b', 'k', 'r']
    # testpairings = [(1, 2), (0, 1), (0, 2)] # The pairings for the TUkey's HSD test

    # Figure save directory etc
    fsd = r'../figures'
    if not os.path.exists(fsd):
        os.makedirs(fsd)

    # FIGURE 1: Barplot of the valences for the given learning rate
    figvpi, axvpi = plt.subplots(figsize=(17, 8))
    figvpi.canvas.manager.full_screen_toggle()
    axvpi.bar([0.5, 1.1, 1.7, 2.3, 3.3, 3.9, 4.5, 5.1], np.mean(valence, axis=0).T.reshape(-1), color=colors * 4, width=0.55)
    axvpi.errorbar([0.5, 1.1, 1.7, 2.3, 3.3, 3.9, 4.5, 5.1], np.mean(valence, axis=0).T.reshape(-1), yerr=np.std(valence, axis=0).T.reshape(-1),
                   fmt='none', ecolor='gray', capsize=10, elinewidth=5, capthick=5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(labels))]
    axvpi.legend(handles, labels, bbox_to_anchor=(1, 1))

    axvpi.set_xticks([1.4, 4.2])
    axvpi.set_xlim([0.15, 5.6])
    axvpi.set_xticklabels([r'$CS^+$', r'$CS^-$'])
    axvpi.set_ylabel('Avoidance index')
    figvpi.subplots_adjust(top=0.978, bottom=0.074, left=0.074, right=0.757, hspace=0.2, wspace=0.2)

    # Save figure
    plt.pause(0.01)
    [figvpi.savefig(os.path.join(fsd, 'fig%s_valence_preference_comparison.'%figname + ext)) for ext in extensions]
    plt.close(figvpi)