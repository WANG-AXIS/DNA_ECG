# DNA_ECG

Code for Decorrelative Network Architecture for Robust Electrocardiogram Classification.

The data ECG data used in this work is from the 2017 PhysioNet Cardiology Challenge (https://physionet.org/content/challenge-2017/1.0.0/).

Raw data (inputs) are excluded from this repo. Download raw_data.npy from: https://drive.google.com/file/d/1u10ZvEilpZnOB4ls5ZAU5ywwf9g23oiB/view?usp=sharing and place in /data before running.

This work borrows scripts and substantial portions of code from Han, Xintian, et al. (https://github.com/XintianHan/ADV_ECG), specifically for preprocessing and loading data, testing models, and crafting both PGD and smoothed adversarial perturbation (SAP) adversarial attacks. The default model (model/best_model.pth) is also the trained model from Han, et al.

Description of files:  
-train: Contains scripts used to train auxillary networks.  
-train/save_features.py: Script used to save features extracted from training data using a particular network (necessary for decorrelation of successive networks).  
-save_pgd_samples.py: Script for creating and saving PGD adversarial samples crafted against a target network.  
-save_pgd_conv_samples.py: Script for creating and saving SAP adversarial samples crafted against a target network.  
-evaluate: Directory that contains scripts for evaluating network ensembles once adversarial samples are generated.
-utils/filters.py: Codes filters used in Fourier partitioning.  
-utils/get_R.py: Codes functions used in feature decorrelation.  
