This repository contains noise spectra for individual pulsars and stochastic gravitational wave background sensitivity curves for the NANOGrav 
15-year data set analysis, highlighted in the paper "The NANOGrav 15-Year Data Set: Detector Characterization and Noise Budget". As in the 
paper these spectra include the noise recovered from a common uncorrelated process analysis across the entire PTA. In other words the spectra 
include the white noise, the power from the common process and any significant additional red noise in an individual pulsar. The three files 
contain the following files of spectra:

* 'characteristic_strain_noise_spectra_NG15yr_psrs.txt' contains a comma separated table of characteristic noise spectra for the 67 pulsars used 
in the 15-year gravitational wave analysis. The header lists the pulsar names in the various columns. The first column is the list of frequencies at 
which the characteristic strain is evaluated.

* 'residual_noise_power_spectra_NG15yr_psrs.txt' contains a comma separated table of residual noise power spectra for the 67 pulsars used in the 15-year gravitational wave analysis. The header lists the pulsar names in the various columns. The first column is the list of 
frequencies at which the power spectral density is evaluated.

* 'sensitivity_curves_NG15yr_fullPTA.txt' contains stochastic gravitational wave background sensitivity curves for the full array of pulsars in 
the PTA. The header lists the contents of the various columns. The first column is the frequencies where the curves were evaluated. The second 
column is the characteristic strain, the third column is the strain power spectral density and the last column is the ratio of cosmological 
energy density, calculated using H0=67.4 km/s/Mpc to match the paper  
