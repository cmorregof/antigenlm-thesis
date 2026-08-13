# Paper 2 outline: Probabilistic latent dynamics in PCA space

## 1. Introduction

- Motivation: forecasting viral evolution requires uncertainty-aware dynamics,
  not only point predictions.
- Problem: full latent SDEs are difficult to justify without a minimal dynamical
  baseline.
- Aim: evaluate whether monthly AntigenLM latent centroids exhibit predictable
  dynamics in low-dimensional PCA space.

## 2. Latent dynamics in influenza

- Influenza A H1N1 and H3N2 as subtype-specific evolutionary systems.
- Rationale for separating subtypes.
- Relationship to antigenic drift, without claiming immune-escape modeling yet.

## 3. PCA state space construction

- Use cached embeddings only.
- Fit PCA within rolling-origin train windows.
- Evaluate d=3, d=4, d=5 reduced state spaces.

## 4. Monthly centroids

- Aggregate records by subtype and year-month.
- Use monthly centroids in PCA space as state variables.
- Track number of records per month.

## 5. Rolling-origin evaluation

- Train up to each origin month.
- Predict the next observed monthly centroid.
- Evaluate 2019-2022.
- Compare against persistence and constant velocity.

## 6. Probabilistic linear dynamics

- Gaussian random walk.
- Linear drift / VAR(1).
- VAR(2).
- Ridge regularization.
- Residual covariance, covariance regularization, and calibration.

## 7. Results

- H1N1 favors persistence/random walk or weak drift.
- H3N2 favors linear drift, especially VAR(2) in seed42.
- Calibration and coverage show tradeoffs between RMSE, NLL, and uncertainty.
- Robustness seed42 vs seed7 supports H3N2 drift more strongly than H1N1.

## 8. Discussion

- The dynamics are subtype-dependent.
- H3N2 appears more amenable to linear latent drift.
- H1N1 may require random-walk models, weaker drift, or more robust sampling
  checks.
- PCA-space dynamics are a stepping stone toward, not a replacement for, a
  biologically interpretable SDE.

## 9. Limitations

- Operates on monthly centroids, not individual sequences.
- Does not decode or generate sequences.
- Does not include explicit immune escape or viability functionals.
- Requires comparison with biological baselines.
- Prospective 2022-2026 validation remains pending.

## 10. Future work toward SDEs

- Extend from discrete Gaussian dynamics to continuous-time SDEs.
- Add subtype-specific drift/diffusion structures.
- Incorporate biological potentials only after baseline dynamics are stable.
- Test prospective validation with a pre-registered protocol.

