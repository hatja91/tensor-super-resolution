# tensor-super-resolution
3D SISR using canonical polyadic and Tucker decompositions

This repository contains the algorithms of the articles
- https://doi.org/10.1109/TMI.2018.2883517 - __TF_SISR.m__
  A Tensor Factorization Method for 3-D Super Resolution With Application to Dental CT
  Authors: Janka Hatvani;Adrian Basarab;Jean-Yves Tourneret;Miklós Gyöngy;Denis Kouamé
- https://doi.org/10.1109/ICIP.2019.8803354 - __TF_SISR_blind.m__
  Tensor-Factorization-Based 3d Single Image Super-Resolution with Semi-Blind Point Spread Function Estimation
  Authors: J. Hatvani; A. Basarab; J. Michetti; M. Gyöngy; D. Kouamé
- https://arxiv.org/abs/2009.08657 (accepted for ISBI 2021) - __TD_SISR.m__
  Single Image Super-Resolution of Noisy 3D Dental CT Images Using Tucker Decomposition 
  Authors: J. Hatvani, A. Basarab, J. Michetti, M. Gyöngy, D. Kouamé
  
For running the codes you will need to download the tensorlab toolbox from https://www.tensorlab.net and add the source codes to the 'tensorlab_2016-03-28' (or latest version) folder.

Links to a sample high- and low-resolution image volume are provided in gt_link.txt and train_link.txt.
