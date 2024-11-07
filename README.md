
<p style="font-size:18px;"> **DNA_ECG** </p>  
This repository contains code to replicate and build upon experiments from the paper "Decorrelative Network Architecture for Robust Electrocardiogram Classification" (preprint: https://arxiv.org/abs/2207.09031).  


**Data Used**  
The data ECG data used in this work is from the 2017 PhysioNet Cardiology Challenge (https://physionet.org/content/challenge-2017/1.0.0/).
Data from the 2018 China Physiological Signal Challenge are also used (http://2018.icbeb.org/Challenge.html).

Raw data (inputs) are excluded from this repo. Download raw_data.npy from: https://drive.google.com/file/d/1u10ZvEilpZnOB4ls5ZAU5ywwf9g23oiB/view?usp=sharing and place in /data before running.  

**Installing the Environment**  
With conda installed, use conda create 

**Training Ensembles**
Description of files:  
-train/train.py: Script used to train ensembles.  
-evaluate_adv.py: Script for saving ensemble inference results against adversarial attacks.  
-utils/filters.py: Codes filters used in Fourier partitioning.  
-utils/decorrelation_func.py: Functions used for feature decorrelation.  

**Testing with Adversarial Attacks**

**Processing Results**


This repository will be updated and a link to the definitive paper will be added once published. 

**Attributions**  
This work borrows and modifies scripts from Han, Xintian, et al. (https://github.com/XintianHan/ADV_ECG), specifically for preprocessing and loading data, and crafting both PGD and SAP adversarial attacks, and defining the network architecture.
