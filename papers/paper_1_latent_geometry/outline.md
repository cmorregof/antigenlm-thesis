# Paper 1 outline: Latent geometry of AntigenLM

## 1. Introduction

- Motivation: protein/genome language models are increasingly used as latent
  state spaces for biological evolution.
- Problem: before modeling continuous dynamics, the geometry of the latent space
  should be audited.
- Case study: AntigenLM embeddings for Influenza A H1N1 and H3N2.
- Main question: does the latent space preserve biologically meaningful
  relationships beyond chronology?

## 2. AntigenLM and latent representations

- Brief description of AntigenLM and its influenza sequence context.
- Embeddings used as fixed representations.
- Scope: this work audits geometry; it does not claim sequence generation or
  vaccine prediction.

## 3. Methods

- Embeddings from cached AntigenLM representations.
- Sampling stratified by year.
- Temporal distance between collection months.
- Hamming distance on HA.
- Hamming distance on HA+NA.
- PCA effective dimension and participation ratio.
- TwoNN intrinsic dimension.
- Local temporal nearest-neighbor analysis.
- Exact HA+NA deduplication.

## 4. Results

- Molecular similarity strongly correlates with latent distance.
- Temporal global correlation is weak.
- Local temporal neighborhoods are coherent.
- Effective and intrinsic dimension appear low.
- PCA visualizations reveal strong anisotropy and subtype organization.

## 5. Discussion

- The latent space appears to encode molecular structure more strongly than
  simple chronological order.
- Local temporal coherence suggests potential usefulness for local dynamics.
- Low effective dimension motivates reduced-state modeling, but does not prove
  that an SDE will work.

## 6. Limitations

- Results depend on the available checkpoint, sampling, and GISAID records.
- Hamming distance is a molecular proxy, not a full antigenic/immunological
  distance.
- TwoNN remains a preliminary estimator and should be complemented by additional
  intrinsic-dimension analyses.
- PCA projections are descriptive and should not be overinterpreted as causal
  trajectories.

## 7. Future work

- Add more seeds and sampling regimes.
- Compare against antigenic cartography or serological data if available.
- Add additional dimension estimators.
- Test whether geometry remains stable under prospective 2022-2026 data.
- Use the audit as a go/no-go step before fitting continuous latent dynamics.

