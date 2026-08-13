# Abstract draft

Continuous stochastic models are attractive for representing viral evolution,
but applying them directly to high-dimensional language-model embeddings requires
evidence that meaningful dynamics exist in the representation space. We evaluate
minimal probabilistic dynamics in a low-dimensional PCA state space derived from
cached AntigenLM embeddings of Influenza A H1N1 and H3N2. For each subtype, we
construct monthly centroids in PCA space and perform rolling-origin evaluation
over 2019-2022, fitting PCA only on data available before each prediction month.
We compare persistence, constant velocity, Gaussian random walk, ridge VAR(1),
and ridge VAR(2) models in dimensions three to five. The results indicate a
subtype-dependent dynamical pattern. For H1N1, persistence and Gaussian random
walk remain difficult to improve upon in the main cache, although a smaller
seed-7 cache shows a modest VAR(2) improvement, suggesting weak or
sampling-sensitive drift. For H3N2, linear drift models consistently improve
over persistence in the main cache, with VAR(2) achieving the strongest RMSE
improvement, and robustness checks preserve evidence for linear drift across
cache/seed changes. Probabilistic evaluation using negative log-likelihood,
Mahalanobis distance, and empirical coverage reveals calibration tradeoffs,
including moderate underdispersion in some accurate H3N2 configurations. These
findings do not constitute full vaccine prediction and do not implement a final
immune-escape SDE. Instead, they establish a minimal, reproducible dynamical
baseline in PCA space and identify H3N2 as the stronger candidate for subsequent
continuous stochastic modeling.

