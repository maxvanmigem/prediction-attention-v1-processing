# prediction-attention-v1-processing
Repository for a study on the effects of prediction and attention on early visual processing (C1, P1, N1 and P3 ERP components).
This repositiory contains everything related to this project such as stimulus presentation and behavioural scripts, data processing scripts, deconvolution and statistical analysis scripts.

Van Migem, M., Marinazzo, D., & Pourtois, G. (2026). Dissociable effects of attention and prediction on visual processing: Evidence from overlap-corrected visual erps. Psychophysiology, 63(1), e70219. https://doi.org/10.1111/psyp.70219


## Team
Maximilien Van Migem, Daniele Marianzzo and Gilles Pourtois

## Setup & Requirements

### Experiment
Python 3.8.18  
- Numpy 1.8.4  
- Pandas 2.0.3  
- Psychopy 2023.2.3


### Analysis
Python 3.9.18  
- Numpy 1.26.0
- Matplotlib 3.8.0  
- mne 1.8.0
- Pandas 2.1.1  
- Seaborn 0.13.2


Julia 1.11.5  
- Unfold  
- UnfoldSim  
- DataFrames

## Data pipeline

### pre_processing
1. localiser_precessing.ipynb -> full preprocessing of localiser data
2. preprocessing_proj1.ipynb -> full preposcessing of main data
3. metadata_adjust.ipynb -> align behavioural data and append to EEG data files
### deconvolution
4. fif_transform.ipynb -> transforms preprocessed data to csv files for every electrode and particpant
5. deconvolution.jl -> performs linear deconvolution and creates rERPs
### data_manipulation
6. localiser_properties.ipynb -> generate file containing localiser peak values used for main analysis
7. erp_extraction.ipynb -> extracts uncorrected ERP mean amplitude values for further analysis
8. erp_measures.ipynb -> extracts overlap corrected ERP mean amplitude values for further analysis
### plotting
9. graph_making.ipynb -> plotting figures used in paper