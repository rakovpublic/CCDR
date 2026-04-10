# Test 01 — RVM ν Coefficient from Joint Cosmological Fit

## Summary
- nu_mean: 0.0091916759
- nu_sigma: 0.00079527957
- nu_95ci: [0.0070638944, 0.0099790394]
- bayes_factor: 2.5274072078546232e+154
- pass_pred: false
- pass_lcdm_disfavored: false
- verdict: not_supported

## Interpretation
The current fit does not satisfy the full Test 01 pass conditions. Bayes factor is decisively against the model by the test's own threshold.

## Run configuration
- walkers: 64
- steps: 30000
- burn_in: 265
- thin: 27
- tau_max: 52.82591434173162
- Pantheon+ rows: 1701
- BAO rows: 13
- Planck prior parameters: ['omega_m', 'h']
- Planck source entry: base/plikHM_TTTEEE_lowl_lowE/base_plikHM_TTTEEE_lowl_lowE.paramnames

## Data sources
- Pantheon+ table URL: https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat
- Pantheon+ covariance URL: https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov
- DESI DR2 BAO mean URL: https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt
- DESI DR2 BAO covariance URL: https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt
- Planck PR3 cosmology zip URL: https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip

## Notes
This standalone script does not import or modify P01_rvm_fit_v2_fixed.py.
The Planck term is derived on the fly as a compressed Gaussian prior in (omega_m, h) from the public Planck PR3 cosmology chains.
The BAO sound horizon is computed with the Eisenstein-Hu approximation using fixed ω_b h² unless you edit the script.
