# SCEIM

This repository contains two standalone MATLAB functions for robust GNSS/INS integrity monitoring.

## Files

```text
BFPA_Bayes_factor_prior_adaptation.m
PFHR_Posterior_fault_hypotheses_reweighting.m
```

## 1. BFPA: Bayes Factor Prior Adaptation

`BFPA_Bayes_factor_prior_adaptation.m` performs a robust Kalman measurement update.

It uses channel-wise innovation information to adapt the measurement covariance. Abnormal measurement channels are automatically down-weighted through an equivalent measurement covariance.


### Inputs

| Input | Description |
|---|---|
| `x_pred` | Predicted state vector |
| `P_pred` | Predicted state covariance |
| `H` | Measurement matrix |
| `z` | Measurement vector |
| `R_nominal` | Nominal measurement covariance |

### Outputs

| Output | Description |
|---|---|
| `x_post` | Updated state vector |
| `P_post` | Updated state covariance |
| `out.R_eff` | Adaptive equivalent measurement covariance |
| `out.E_t` | Channel-wise normal-branch probability |
| `out.E_lambda` | Channel-wise equivalent precision |
| `out.kappa` | Final channel weight |
| `out.BF_kj` | Bayes factor for each channel |
| `out.u_kj` | Normalized innovation |
| `out.converged` | Convergence flag |
| `out.iterations` | Number of iterations |


## 2. PFHR: Posterior Fault Hypotheses Reweighting

`PFHR_Posterior_fault_hypotheses_reweighting.m` reweights monitored fault hypotheses using current subset evidence.

It keeps the total monitored fault risk unchanged, and only changes the relative weights among monitored fault hypotheses.


### Inputs

| Input | Description |
|---|---|
| `pap_subset` | Prior hypothesis weights |
| `subset_log_evidence` | Log evidence for each hypothesis |
| `use_posterior` | `1` enables reweighting, `0` keeps original weights |

The first element of `pap_subset` is the no-fault hypothesis. The remaining elements are monitored fault hypotheses.

### Outputs

| Output | Description |
|---|---|
| `p_fault` | Fault weights for protection level computation |
| `posterior_hypothesis` | Posterior probability over valid hypotheses |
| `fault_relative_posterior` | Relative posterior weights among monitored faults |
| `diagnostics` | Validity checks and risk-mass information |


## Notes

Dataset: [IPNL-POLYU](https://github.com/IPNL-POLYU)  
GNSS/INS datasets for performance evaluation under different scenarios.

GNSS/INS Tightly Coupled Framework: [TightlyCoupledINSGNSS](https://github.com/benzenemo/TightlyCoupledINSGNSS.git)  
Provides tightly-coupled integration of GNSS and INS, improving robustness in challenging environments.

ARAIM (via MAAST): [Stanford GPS Lab – MAAST](https://gps.stanford.edu/resources/software-tools/maast)  
Used for simulation, integrity monitoring, and performance testing.
