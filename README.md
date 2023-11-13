# DNA_ECG

Code for Decorrelative Network Architecture for Robust Electrocardiogram Classification.

The data ECG data used in this work is from the 2017 PhysioNet Cardiology Challenge (https://physionet.org/content/challenge-2017/1.0.0/).
Data from the 2018 China Physiological Signal Challenge are also used (http://2018.icbeb.org/Challenge.html).

Raw data (inputs) are excluded from this repo. Download raw_data.npy from: https://drive.google.com/file/d/1u10ZvEilpZnOB4ls5ZAU5ywwf9g23oiB/view?usp=sharing and place in /data before running.

This work borrows and modifies scripts from Han, Xintian, et al. (https://github.com/XintianHan/ADV_ECG), specifically for preprocessing and loading data, and crafting both PGD and SAP adversarial attacks, and defining the network architecture.

Description of files:
-train/train.py: Script used to train ensembles.  
-evaluate_adv.py: Script for saving ensemble inference results against adversarial attacks.
-utils/filters.py: Codes filters used in Fourier partitioning.  
-utils/decorrelation_func.py: Functions used for feature decorrelation.

This repository will be updated and a link to the definitive paper will be added once published. 
