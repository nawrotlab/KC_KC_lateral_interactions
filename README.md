# Modelling part of Manoim-Wolkowitz, Tunc et al.

This repository includes the code for the mushroom body model used for investigating the role of lateral Kenyon cell - Kenyon cell interactions in mediating efficiency and specificity of learning.

## Repository structure

Folder "codes" contains all the scripts, and folder "data" contains calcium imaging data from Manoim et al. 2022 (https://doi.org/10.1016/j.cub.2022.09.007), which is required for fitting model parameters. 
Running the scripts generates further figures used in the manuscript, as well as data files for model simulations, which are not part of this repository. requirements.txt shows the required packages to run the scripts.

Some scripts need to be executed before others:

* **fit_rate_model_to_data.py:** This script is required to be executed BEFORE any other script, as it saves the model parameters fitted to calcium imaging data. Other scripts cannot work without running this script before.
* **rate_model_get_summed_KC_activity_different_model_variants.py:** This script needs to be run before any script that is needed for learning simulations (i.e. learning_rate_screening_mbon_normalized.py AND plot_valences_normalized_mbon),
since it saves the parameters required for tuning the MBON output.
* plot_valences_normalized_mbon needs.py to be run AFTER running learning_rate_screening_mbon_normalized.py, since it depends on the data saved in the latter script.
* plot correlation_different_overlaps.py needs to be run AFTER running inter_odor_activity_correlation_sparseness.py, since it depends on the data saved in the latter script.
* **Function scripts:**
  * fit_functions.py contains all the functions required for fitting model parameters to calcium imaging data.
  * KC_population_calcium_rate_model_functions.py contains all the functions required for setting up  and simulating the model.
  * plotting_related_functions.py is imported in scripts to make figure layouts more aesthetic.

## Reproducing the figures

* **fit_rate_model_to_data.py:** Figures 2 B and S1 F,G
* **plot_valences_normalized_mbon.py:** Figures 2 C and S1 E
* **plot_correlation_different_overlaps.py:** Figure S1 A
* **inter_odor_activity_correlation_sparseness.py:** Figure S1 B-D
* **learning_rate_screening_mbon_normalized.py:** Figure S1 H
* **rate_model_odor_stimulation_activity_histograms.py** Figure S3 E

### contact: itunc@uni-koeln.de, ibrahimalperentunc@protonmail.com
