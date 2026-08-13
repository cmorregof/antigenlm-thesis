# Pre-Submission Reviewer Simulation

Date: 2026-05-05

Manuscript reviewed: `papers/paper_1_latent_geometry_full/main.tex`

## Reviewer 1: Computational Biology / Bioinformatics Methods

### Likely Overall Recommendation

Major revision if submitted to BMC Bioinformatics or Scientific Reports. Rejection if submitted to Bioinformatics without a stronger method, software package, or benchmark.

### Top 5 Criticisms

1. The study is currently a one-model, one-dataset audit rather than a general method or benchmark.
2. Spearman correlations are useful but too limited for metric validation; there are no kNN retrieval, local rank-preservation, trustworthiness/continuity, or distance-correlation analyses.
3. No negative controls are included: random embeddings, date permutations, shuffled labels, matched-spectrum embeddings, or untrained-checkpoint embeddings.
4. Exact duplicates are removed for TwoNN/local temporal analyses but not for the main pairwise molecular correlations.
5. Reproducibility is local rather than publication-grade; software versions, exact command logs, and data-access reconstruction instructions are incomplete.

### What Would Trigger Rejection

- Claiming broad biological or forecasting validity from the current audit.
- Treating Hamming correlations as validation of antigenic similarity.
- Submitting without clarifying checkpoint provenance and data availability.
- Presenting p-values from sampled pairs as decisive statistical evidence while ignoring dependence among distances.

### What Would Make the Paper Compelling

- A small but rigorous robustness bundle:
  - deduplicated Spearman correlations;
  - date-permutation control for local temporal coherence;
  - random or matched-spectrum embedding controls;
  - kNN molecular retrieval precision.
- A clean reusable audit workflow that can be applied to other sequence-model embeddings.
- Public aggregate outputs and rebuild instructions for authorized GISAID users.

### Precise Edits or Analyses to Address Concerns

- Add a subsection "Negative controls" with date-permuted temporal labels and random embeddings.
- Add a table reporting deduplicated Spearman correlations next to full-cache correlations.
- Add kNN retrieval curves: mean HA/HA+NA Hamming distance among top-k latent neighbors versus subtype-matched random and year-matched random baselines.
- Add software versions and exact commands to Methods or supplement.
- Add a paragraph explaining why p-values are omitted or de-emphasized for sampled pairwise correlations.

## Reviewer 2: Viral Evolution / Influenza Biology

### Likely Overall Recommendation

Major revision for Virus Evolution; minor-to-major revision for a computational journal if the paper keeps its audit framing. Rejection for any journal if it implies antigenic or vaccine relevance without external validation.

### Top 5 Criticisms

1. Molecular Hamming distance is not antigenic distance; the manuscript must be very careful not to conflate them.
2. No phylogenetic tree, clade labels, antigenic assays, HI titers, or vaccine-strain outcomes are used.
3. Temporal neighborhood enrichment may reflect sampling bursts, geographic clustering, or surveillance intensity rather than evolutionary state structure.
4. H1N1 and H3N2 have different evolutionary dynamics, but the biological interpretation remains fairly generic.
5. HA and NA are analyzed jointly, but the biological roles of HA and NA are not disentangled; NA-only analysis is absent.

### What Would Trigger Rejection

- Statements that the latent space captures antigenic drift without antigenic data.
- Claims that the embeddings are useful for vaccine-strain recommendation.
- Ignoring GISAID sampling bias and access/acknowledgement requirements.
- Treating collection date as a clean evolutionary clock.

### What Would Make the Paper Compelling

- Validation against Nextstrain clades or curated lineage labels.
- Phylogenetic distance comparison within subtype.
- Antigenic cartography or HI distance comparison for a subset of H3N2/H1N1 records.
- Temporal-neighborhood enrichment stratified by year or major clade.
- Separate HA, NA, and HA+NA analyses with biological interpretation.

### Precise Edits or Analyses to Address Concerns

- Add a stronger paragraph in Limitations: "Molecular similarity is not antigenic similarity."
- Add NA-only Hamming correlations and amino-acid HA/NA correlations.
- Add year-matched random baselines for local temporal neighborhoods.
- Add date-permutation controls within subtype.
- If clade labels are available, report nearest-neighbor clade purity or clade retrieval precision.

## Reviewer 3: Mathematical / Statistical Geometry and Manifold Learning

### Likely Overall Recommendation

Major revision. The paper is promising as an empirical geometry audit but not yet rigorous enough for strong claims about latent-space geometry or intrinsic dimension.

### Top 5 Criticisms

1. PCA effective dimension and TwoNN intrinsic dimension estimate different objects but could still be overinterpreted.
2. TwoNN assumptions are fragile under nonuniform density, anisotropy, duplicate/near-duplicate clusters, and boundary effects.
3. Euclidean distance in raw latent space is taken as given; no whitening, cosine distance, Mahalanobis distance, or PCA-subspace sensitivity is tested.
4. Spearman pair correlations do not establish local geometry preservation.
5. No uncertainty quantification is provided for key local-neighborhood summaries beyond random-seed variation.

### What Would Trigger Rejection

- Claiming a specific intrinsic dimension as a fact.
- Claiming a smooth manifold without estimator agreement and model checks.
- Equating PCA visualization with latent geometry.
- Ignoring density heterogeneity and sampling effects.

### What Would Make the Paper Compelling

- Explicit distinction among linear effective dimension, local intrinsic dimension, and visualization dimension.
- Multiple intrinsic-dimension estimators with sensitivity over k/sample/time/subtype.
- Local neighborhood-preservation metrics between molecular-distance space and latent space.
- Whitened/PCA-subspace sensitivity analyses.
- Bootstrap confidence intervals for temporal-neighborhood enrichment.

### Precise Edits or Analyses to Address Concerns

- Add local PCA or Levina-Bickel MLE as a complementary intrinsic-dimension estimator.
- Recompute molecular correlations using raw Euclidean distance, PCA-whitened Euclidean distance, cosine distance, and top-PC distances.
- Add trustworthiness/continuity or co-ranking matrix diagnostics.
- Add bootstrapped confidence intervals for median neighbor/random ratios.
- Replace any "manifold" language with "low effective structure" unless new evidence is added.

## Cross-Reviewer Synthesis

The revised manuscript is much safer than the initial generated draft because it now frames the study as a geometric audit and explicitly avoids antigenic, vaccine, and forecasting claims. The remaining risk is that the paper may still look under-validated for a journal submission unless one or two controls are added.

The most cost-effective additions are:

1. Date-permutation control for local temporal neighborhoods.
2. Exact HA+NA-deduplicated Spearman correlations.
3. NA-only and amino-acid HA/NA distance analyses.
4. kNN molecular retrieval precision against subtype-matched and year-matched baselines.
5. Software-version and data-rebuild documentation.

## Simulated Editorial Decision

Current revised manuscript, without new analyses:

- BMC Bioinformatics: major revision, potentially acceptable after robustness controls.
- Scientific Reports: major revision or borderline, depending on reviewer appetite for a descriptive audit.
- Virus Evolution: likely rejection or major revision unless phylogenetic/clade/antigenic validation is added.
- Bioinformatics: likely rejection because the manuscript is not yet a method, package, or benchmark.
- PLOS Computational Biology: likely rejection because the biological insight is not yet deep enough.
- Algorithms for Molecular Biology: major revision if reframed as a reusable audit framework.
