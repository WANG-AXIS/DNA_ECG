
# **DNA_ECG**  
This repository contains code to replicate and build upon experiments from the paper "Decorrelative Network Architecture for Robust Electrocardiogram Classification" (preprint: https://arxiv.org/abs/2207.09031).  


## **Data Used**  
The data ECG data used in this work is from the 2017 PhysioNet Cardiology Challenge (https://physionet.org/content/challenge-2017/1.0.0/).
Data from the 2018 China Physiological Signal Challenge are also used (http://2018.icbeb.org/Challenge.html).

Raw data (inputs) are excluded from this repository. Download ```raw_data.npy``` from: https://drive.google.com/file/d/1u10ZvEilpZnOB4ls5ZAU5ywwf9g23oiB/view?usp=sharing and place in ```data/``` before running.  

## **Installing the Environment**  
- Clone this repository.
- Use Conda and ```requirements.txt``` to create a conda environment  
  ```conda create --name <env> --file requirements.txt ```

## **Training and Evaluating Ensembles**
-To train an ensemble run ```python train.py``` in train/. Hyperparameters/settings at the beginning of this script can be modified.  
-To evaluate a trained ensemble, run ```python evaluate_adv.py```. Settings (e.g. directory for ensemble) must be inputted at the beginning of this script.  
-Files in ```results/``` are used to extract and visualize results from a trained ensembles.

Other noteworthy files include:  
-```utils/filters.py```: Codes filters used in Fourier partitioning.  
-```utils/decorrelation_func.py```: Functions used for feature decorrelation.  

This repository will be updated and a link to the definitive paper will be added once published. 

## **Acknowledgments**  
This work borrows and modifies scripts from Han, Xintian, et al. (https://github.com/XintianHan/ADV_ECG), specifically for preprocessing and loading data, and crafting both PGD and SAP adversarial attacks, and defining the network architecture.
