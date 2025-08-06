#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import plotting_related_functions
import os
import pandas as pd
import scipy.stats as st

# Plot the correlation of the KD model with different overlaps
extensions = ['png', 'pdf'] # Figure save extension
fsd = r'../figures'
if not os.path.exists(fsd):
    os.makedirs(fsd)

# Load data
ldir = r'../data/model_related/overlap_correlation_sparseness'
files = os.listdir(ldir)
overlaps = ['.'.join(f.split('.')[:-1]).split('_')[-1] for f in files if f.endswith('.csv')]
avg_std_ovrlps = np.zeros([len(overlaps), 2])
corr_merge = []
ovrlp_merge = []
# Slice the dataframe for the relevant part -> Extract only correlation with the KD model for each case, also separately get the unique random_overlap entries
for i, (file, overlap) in enumerate(zip(files, overlaps)):
    data = pd.read_csv(os.path.join(ldir, file))
    # Remove the unnecessary columns
    data = data[data['Model'] == 'KD']
    # Extract CS +/- overlaps and sort them into a 2D array
    grouped_arrays = [v.to_numpy() for _, v in data.groupby('Odor')[['Overlap']]]
    ovrl = np.hstack(grouped_arrays) # CS+/- percent random_overlap array, shape n_instances x n_odors (CS+/- in this order)
    # Calculate the average and std random_overlap for each odor
    avg_std_ovrlps[i] = (np.mean(ovrl), np.std(ovrl))
    # Extract the correlation values per instance to a n_instances array, remove the redundant entry for both odors
    corr = data[data['Odor']=='CS+'].sort_values('Instance')['Correlation'].to_numpy()
    corr_merge.append(corr)
    ovrlp_merge.append(ovrl.mean(axis=1))
# Flatten the data_merge list to a single array
corr_merge = np.concatenate(corr_merge)
ovrlp_merge = np.concatenate(ovrlp_merge)
# Linear regression
slope, intercept, r_value, p_value, std_err = st.linregress(ovrlp_merge, corr_merge)
# See what random_overlap is best fitting for the 0.2 correlation
best_ovrl = (0.2 - intercept)/slope # Overlap fitting best to data
print(f'{best_ovrl:.2f} % random_overlap of KC responders required for an average correlation of 0.2 between 2 odors. \n')

# Figure
fig, ax = plt.subplots(figsize=(5.7, 5.8))
fig.subplots_adjust(top=0.955, bottom=0.16, left=0.245, right=0.97, hspace=0.2, wspace=0.2)
# Scatter plot
ax.scatter(ovrlp_merge, corr_merge, color='black', alpha=0.5)
# Plot regression line
ax.plot([ovrlp_merge.min(), ovrlp_merge.max()], [intercept + slope * ovrlp_merge.min(), intercept + slope * ovrlp_merge.max()], color='red', lw=2)
ax.text(0.7, 0.2, f'R² = {r_value**2:.2f}', ha='center', va='center',transform=ax.transAxes)
ax.set_xlabel('Overlap [%]')
ax.set_ylabel('Correlation')
plt.pause(0.01)
# Save figure
[fig.savefig(os.path.join(fsd, f'figs2a_correlation_overlap.{extension}')) for extension in extensions]
