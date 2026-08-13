# Scientific Review and Missing Tests

Date: 2026-05-05

Scope: pre-submission review of `papers/paper_1_latent_geometry_full/main.tex`, focused on the geometry of cached AntigenLM latent representations for Influenza A HA/NA records.

## Executive Assessment

The current paper has the core of a defensible computational audit: a large local embedding cache, explicit subtype stratification, molecular-distance correlations, PCA spectra, TwoNN sensitivity, exact HA+NA deduplication for local analyses, and a clear refusal to claim generation, vaccine recommendation, or prospective forecasting.

The paper is not yet a strong international-journal submission because it underdevelops the validation logic. A reviewer will ask: Why is Euclidean distance the right metric? Is Hamming distance an adequate biological proxy? Are local temporal neighborhoods simply caused by sampling density or duplicate-like clusters? Do PCA and TwoNN actually justify a low-dimensional manifold claim? What controls show that the observed structure is not a trivial consequence of subtype/year imbalance or embedding anisotropy? The present repository supports a descriptive audit, not a biological validation of antigenicity, forecasting, or evolutionary mechanism.

The prudent target after revision is a computational analysis paper, not a forecasting or vaccine-strain paper.

## Major Strengths

- Large analysis scale: `111,756` cached embeddings from H1N1 and H3N2 records is substantial for a local geometric audit.
- Clear local representation definition: embeddings are 384-dimensional mean-pooled hidden states from the local prediction checkpoint.
- Subtype stratification: pairwise correlations are estimated within H1N1 and H3N2, avoiding the most obvious artifact of mixing subtypes.
- Molecular proxy validation: strong Spearman correlations against HA and HA+NA normalized Hamming distances support molecular organization.
- Duplicate-aware local analyses: TwoNN and temporal-neighborhood diagnostics remove exact HA+NA duplicates.
- Multiple seeds: pair sampling and TwoNN sensitivity use multiple random seeds.
- Honest scope boundary: the draft already disclaims generation, mutation optimization, vaccine recommendations, and forecasting reproduction.

## Major Weaknesses

- The validation framework is narrow. Spearman correlation alone is insufficient to establish metric validity, neighborhood fidelity, or downstream state-space usability.
- The biological proxy is limited to nucleotide Hamming distance. No amino-acid, codon-aware, protein-substitution-aware, antigenic, phylogenetic, or clade-based validation is present.
- Temporal structure is only tested globally by pairwise month difference and locally by subtype-matched random baseline. This is useful but not enough to exclude sampling bias, year imbalance, or density artifacts.
- Intrinsic dimension is supported by PCA and TwoNN only. Both are sensitive to anisotropy, density heterogeneity, finite-sample effects, duplicate/near-duplicate structure, and preprocessing.
- The paper has no negative controls: no random embeddings, permuted dates, random projections, shuffled labels, PCA-whitened distances, or untrained-checkpoint baseline.
- AntigenLM provenance and citation metadata are not yet verified.
- The discussion does not yet engage deeply with the distinction between molecular similarity, antigenic similarity, evolutionary relatedness, and predictive fitness.
- The current manuscript voice is still that of a thesis checkpoint rather than a polished journal article.

## Essential Revisions Before Submission

1. Reframe the paper as an embedding audit, not as evidence that AntigenLM latent space is biologically meaningful in a broad sense.
2. Replace strong language such as "preserves biological similarity" with "preserves molecular similarity under HA/HA+NA Hamming proxies."
3. Explain why weak global temporal correlation is expected under branching evolution, recurrent sampling, and nonlinear drift; do not treat time as a single latent axis.
4. Interpret local temporal coherence as neighborhood enrichment, not forecasting skill.
5. State that antigenic similarity is not measured because no HI assay, neutralization, or antigenic cartography data are included.
6. State that PCA effective dimension, participation ratio, and TwoNN estimate different properties and should not be collapsed into one "dimension" claim.
7. Add specific limitations on TwoNN: duplicates, anisotropy, nonuniform density, boundary effects, finite-sample effects, and estimator sensitivity to trimming/standardization.
8. Add a reproducibility and data/code availability section that acknowledges likely GISAID restrictions and distinguishes raw data, cached embeddings, aggregate metrics, figures, and scripts.
9. Add a responsible-use statement: no sequence generation, no mutation optimization, no vaccine or public-health recommendation.
10. Verify AntigenLM citation and checkpoint/source metadata before citing it as a published model.

## Essential Analyses Before a Strong Submission

These are not currently supported by repository outputs. They should either be run or explicitly listed as future work.

- Bootstrap confidence intervals for Spearman correlations and temporal-neighborhood summaries, ideally resampling records rather than only pair draws.
- Date-permutation control: permute years/months within subtype and rerun local temporal-neighbor enrichment to show the signal depends on true temporal labels.
- Random-embedding control: compare pairwise and kNN metrics against Gaussian random embeddings with the same dimensionality and optionally with matched PCA spectrum.
- PCA-whitened control: recompute molecular/temporal correlations and local kNN after whitening or after using top-k PCs to test sensitivity to anisotropy.
- NA-only distance: report latent distance versus NA Hamming separately to determine whether HA+NA improvement comes from NA or length-weighted averaging.
- Amino-acid distance: translate HA/NA and compute amino-acid Hamming for in-frame sequences, with careful handling of ambiguous bases and indels.
- kNN retrieval validation: for each query, ask whether nearest latent neighbors are enriched for low molecular distance relative to subtype-matched candidates.
- Local correlation by latent-distance quantile: test whether molecular-distance alignment is strongest locally, which is more relevant for state-space modeling than global rank correlation.
- Trustworthiness/continuity or rank-preservation metrics: quantify neighborhood preservation between molecular-distance space and latent space.
- Year- or month-matched baseline for molecular similarity: compare nearest neighbors against random candidates from similar collection times to separate temporal density from sequence similarity.
- Length-matched baseline: ensure distance correlations are not driven by sequence length differences or filtering.
- Exact duplicate exclusion for pair correlations: rerun pairwise Spearman after exact HA+NA deduplication, or at minimum quantify the impact of duplicates on molecular rho.

## Optional but Valuable Experiments

- Mantel-style test between sampled latent-distance and molecular-distance matrices within subtype. Use cautious interpretation because distance entries are dependent.
- Distance correlation or HSIC between latent distance and molecular/temporal metrics, with permutation controls.
- Local intrinsic dimension by subtype and by time window, using complementary estimators such as Levina-Bickel MLE, correlation dimension, local PCA, or DANCo if computationally feasible.
- Clade or lineage classification if trusted clade labels are available. This would strengthen the evolutionary interpretation without requiring antigenic assays.
- Phylogenetic distance validation if trees can be built or obtained. This would be highly valuable for Virus Evolution.
- Antigenic cartography/HI distance validation if assay data are available. This is essential for any claim about antigenic similarity.
- Geographic/host-stratified sensitivity analyses to address GISAID sampling bias.
- Comparison to simpler sequence embeddings such as k-mer TF-IDF, one-hot PCA, or protein language model embeddings if available.
- Comparison to an untrained or randomly initialized checkpoint with identical architecture, if feasible.
- Stability across alternative pooling strategies: mean pooling, last-token pooling, HA-token pooling, NA-token pooling, and potentially [segment]-specific pooling if the architecture supports it.

## A. Metric Validity

### Current Evidence

The current paper supports a clear result: Euclidean distances in the local 384-dimensional embedding space are strongly rank-associated with nucleotide Hamming distances for HA and HA+NA within subtype. The association is stronger for H1N1 than H3N2 and stronger for HA+NA than HA-only in both subtypes.

### What This Establishes

- The embedding geometry is not arbitrary with respect to the input sequences.
- Nearby/farther points in latent Euclidean distance tend to correspond to more similar/dissimilar HA or HA+NA nucleotide sequences under the current Hamming implementation.
- The model likely encodes substantial molecular variation in its hidden-state representation.

### What This Does Not Establish

- It does not prove Euclidean distance is the optimal latent metric.
- It does not prove local neighborhoods are biologically meaningful beyond molecular similarity.
- It does not validate antigenic similarity, immune escape, fitness, clade identity, or forecasting value.
- It does not show that the relationship is equally strong at local and global scales.
- It does not rule out confounding by time, length, sampling density, or subtype-specific structure.

### Is Spearman Correlation Enough?

No. Spearman correlation is a useful first audit because it tests monotonic rank association and is robust to nonlinear scaling. It is not sufficient because:

- pairwise distances are dependent observations;
- a high global rank correlation can coexist with poor local-neighborhood preservation;
- a low global temporal correlation can coexist with strong local temporal coherence;
- correlation does not test retrieval precision, local geometry, or state-space continuity;
- correlation does not identify whether the signal is driven by a few dominant axes or clusters.

### Recommended Additions

- kNN molecular retrieval: mean/median Hamming distance among top-k latent neighbors compared with subtype-matched random, year-matched random, and length-matched random baselines.
- Trustworthiness/continuity between molecular-distance neighborhoods and latent-distance neighborhoods.
- Local Spearman correlation by latent-distance quantile or molecular-distance quantile.
- Distance correlation or HSIC with permutation testing.
- Mantel-style tests as supplementary diagnostics, with a note about dependency and permutation assumptions.
- Compare HA-only, NA-only, unweighted HA+NA, and length-weighted HA+NA distances.
- Add amino-acid Hamming and, if feasible, substitution-aware protein metrics. Nucleotide Hamming is a blunt proxy because synonymous mutations and amino-acid substitutions differ biologically.

## B. Temporal and Evolutionary Structure

### Current Evidence

The manuscript supports two temporal findings:

- global latent distance has weak correlation with absolute month difference;
- latent nearest neighbors after exact HA+NA deduplication are much closer in collection time than subtype-matched random pairs.

### Interpretation

Weak global temporal correlation is expected under influenza evolution. Evolutionary trajectories branch, reassort, experience changing selection pressures, include co-circulating lineages, and are sampled unevenly across time and geography. Absolute chronological distance is not the same as evolutionary distance.

Local temporal coherence is more relevant to a candidate state-space argument. If nearby latent points are usually close in time, then local neighborhoods may represent recent molecular states or densely sampled portions of ongoing evolutionary trajectories. This is still descriptive and hypothesis-generating.

### Risks

- Sampling bias: GISAID sampling intensity varies strongly by year, geography, and outbreak context.
- Density artifact: if many similar sequences are deposited in bursts, nearest neighbors will be temporally close even without deeper evolutionary structure.
- Duplicate-like records: exact duplicates are removed, but near-duplicates remain.
- Temporal leakage: using the full data cloud to learn PCA or inspect geometry can blur retrospective/prospective framing if later used for forecasting claims.
- Subtype differences: H1N1 and H3N2 have different epidemiological and evolutionary dynamics; combining them too casually is risky.

### Recommended Additions

- Date-permutation control within subtype.
- Year-matched and length-matched random baselines.
- Compare nearest-neighbor temporal enrichment before and after exact HA+NA deduplication; the current manuscript only foregrounds the deduplicated full-data result.
- Compute enrichment for thresholds: proportion of top-k neighbors within 1, 3, 6, 12 months versus random baselines.
- Report results separately by subtype and possibly by time periods with sufficient sample size.
- If available, validate against clade labels or phylogenetic distances.
- Avoid any claim of "evolutionary trajectory" unless supported by phylogenetic or lineage evidence.

## C. Intrinsic Dimension and Manifold Structure

### Current Evidence

The paper reports:

- very low PCA effective dimension, with global `n90=2`, `n95=3`, and participation ratio `1.91`;
- TwoNN estimates around `3.85-3.90` with trim `0.01` and `5.40-5.50` with trim `0.05`, across sample sizes.

### Correct Interpretation

PCA effective dimension quantifies linear variance concentration, not manifold dimension. A low participation ratio indicates strong anisotropy and dominance by a few variance directions. It can arise from subtype separation, temporal drift, batch effects, model anisotropy, or true low-dimensional biological variation.

TwoNN estimates a local intrinsic dimension under assumptions about locally uniform sampling on a manifold. It is useful because it does not simply read off the PCA spectrum, but it is sensitive to data density, duplicates, anisotropy, boundary effects, standardization, and trimming.

The correct paper-level claim is:

> PCA and TwoNN provide complementary evidence that the cached embedding cloud has strong low-dimensional structure under the tested preprocessing choices.

The manuscript should not claim:

> The latent space is a 4-dimensional manifold.

### Recommended Additions

- Repeat TwoNN by subtype after deduplication.
- Report local PCA dimension or participation ratio by subtype/time window.
- Add Levina-Bickel MLE or another kNN intrinsic-dimension estimator with sensitivity across k.
- Add a random-projection or PCA-whitened sensitivity analysis.
- Discuss that PCA dimension, TwoNN dimension, and visualization dimension are distinct.
- If the paper keeps the phrase "manifold," use it cautiously and possibly as "manifold-like low-dimensional structure" rather than a mathematical manifold claim.

## D. Robustness and Controls

### Current Strengths

- Pair-sampling seeds are varied.
- TwoNN sample size, trim, and seeds are varied.
- Exact HA+NA deduplication is used for TwoNN and local temporal-neighbor analyses.

### Missing Controls

- Exact duplicate removal for pairwise Spearman correlations.
- Date-permutation control.
- Random embeddings.
- Matched-spectrum random embeddings.
- PCA-whitened embeddings.
- Random projections of original embeddings.
- Shuffled subtype/year labels.
- NA-only distances.
- Amino-acid distances.
- Bootstrap CIs.
- Sampling-strategy sensitivity beyond the current full cache.
- Sensitivity to geographic/host/year imbalances.

### Essential Robustness Recommendation

At minimum, add three controls before submission if time permits:

1. Date-permutation control for local temporal coherence.
2. Exact HA+NA-deduplicated Spearman correlations.
3. NA-only and amino-acid HA/NA Hamming distances.

These would materially strengthen the paper while staying within the current audit framing.

## E. Biological Interpretation

### Defensible Claims

- The representation organizes HA/NA records by molecular sequence similarity under the tested Hamming proxies.
- The representation contains local temporal coherence after removing exact HA+NA duplicates.
- The embedding cloud has strong low-dimensional variance concentration and low estimated local dimension under PCA/TwoNN diagnostics.
- These properties make the representation worth considering as a candidate state space for future dynamical modeling.

### Claims Requiring Softening

- "Biologically meaningful latent space" should be replaced with a precise claim: "molecularly organized under HA/HA+NA Hamming proxies."
- "Evolutionary state space" should be softened to "candidate state space for future evolutionary modeling."
- "Temporal organization" should be split into "weak global chronological association" and "strong local temporal-neighborhood enrichment."
- "Low-dimensional manifold" should be softened to "low effective dimension under complementary diagnostics."

### Unsupported Claims

- Antigenic similarity.
- Immune escape.
- Vaccine-strain relevance.
- Predictive fitness.
- Prospective forecasting.
- Sequence generation quality or safety.
- Phylogenetic validity.
- Clade/lineage resolution.

### Evidence Needed for Stronger Biological Claims

- Antigenic cartography or HI assay distances for antigenic similarity.
- Phylogenetic trees or patristic distances for evolutionary-distance validation.
- Nextstrain or WHO clade labels for lineage/clade structure.
- Prospective time-split evaluation for forecasting.
- Vaccine-strain and post-vaccine-season outcomes for vaccine relevance.
- Functional assays or curated mutation annotations for immune escape.

## F. Journal-Level Contribution

### Current Identity of the Paper

The current paper is closest to a reproducible computational data-analysis paper or embedding-audit case study. It is not yet a methods paper unless the audit is formalized as a reusable framework. It is not yet a strong viral evolution paper because it lacks phylogenetic, clade, and antigenic validation. It is not yet a forecasting paper.

### Journal Fit Ranking

1. **BMC Bioinformatics**
   - Best current fit.
   - Strongest angle: transparent computational audit of sequence-language-model latent geometry for influenza HA/NA embeddings.
   - Missing before submission: stronger reproducibility section, citation cleanup, controls or explicit limitations.

2. **Scientific Reports**
   - Plausible broad-scope fit if the paper is polished and honest.
   - Strongest angle: large-scale descriptive analysis of learned influenza representations.
   - Missing before submission: stronger controls and clearer biological framing.

3. **Algorithms for Molecular Biology**
   - Possible if reframed as a general embedding-audit workflow with reproducible diagnostics.
   - Missing before submission: algorithmic contribution, reusable pipeline abstraction, comparison to baselines.

4. **Virus Evolution**
   - Currently weaker fit.
   - Strongest angle would require linking latent neighborhoods to viral evolutionary processes.
   - Missing before submission: clade labels, phylogenetic distances, phylodynamic context, antigenic drift interpretation.

5. **Bioinformatics**
   - Ambitious and currently not the right fit.
   - Missing before submission: method/software contribution, benchmark, reusable package, comparison across models/datasets.

6. **PLOS Computational Biology**
   - Ambitious and not currently justified.
   - Missing before submission: broader biological insight, stronger mechanistic interpretation, external validation, forecasting or antigenic relevance.

## Recommended Target Journals Ranked by Fit

1. BMC Bioinformatics
2. Scientific Reports
3. Algorithms for Molecular Biology
4. Virus Evolution
5. Bioinformatics
6. PLOS Computational Biology

## Recommended Title Variants

### BMC Bioinformatics Angle

- "Auditing the Geometry of AntigenLM Latent Representations for Influenza A HA and NA Sequences"
- "A Reproducible Geometric Audit of Influenza A Sequence-Language-Model Embeddings"
- "Molecular and Temporal Structure in AntigenLM Embeddings of Influenza A HA/NA Records"

### Scientific Reports Angle

- "Large-scale Geometric Analysis of AntigenLM Representations for Influenza A HA and NA"
- "Molecular Organization and Temporal Locality in Learned Influenza A Sequence Representations"
- "Low-dimensional Structure in AntigenLM Latent Embeddings of Influenza A Surface Genes"

### Algorithms for Molecular Biology Angle

- "A Geometry-First Framework for Auditing Biological Sequence-Model Embeddings"
- "Quantifying Metric, Temporal, and Intrinsic-Dimension Structure in Viral Sequence Embeddings"
- "Reproducible Diagnostics for Learned Latent Spaces of Viral HA/NA Sequences"

### Virus Evolution Angle

- "Local Temporal Coherence in Learned Representations of Influenza A HA/NA Evolution"
- "Latent Neighborhood Structure of Influenza A Surface Genes in AntigenLM Embeddings"
- "Molecular Similarity and Temporal Locality in Influenza A Sequence-Model Embeddings"

### Bioinformatics Angle

- "Benchmarking Geometric Fidelity of Biological Sequence Language Model Embeddings"
- "A Reusable Audit Pipeline for Metric and Manifold Structure in Protein Language Model Embeddings"
- "Quantitative Geometry Diagnostics for Viral Sequence Representation Learning"

### PLOS Computational Biology Angle

- "What Learned Influenza Representations Encode About Molecular Similarity and Evolutionary Time"
- "Geometry of Viral Sequence Language Model Embeddings Reveals Molecular Organization and Temporal Locality"
- "Interpreting Learned State Spaces for Influenza A Evolution"

## Claims That Must Be Softened

- "Latent distance preserves biological similarity" -> "latent distance is strongly associated with HA/HA+NA molecular Hamming distance."
- "The embedding space is biologically meaningful" -> "the embedding space is molecularly organized under the tested proxies."
- "The latent space is a low-dimensional manifold" -> "the embedding cloud shows low effective dimension under PCA and low estimated local dimension under TwoNN."
- "Temporal structure is strong" -> "global temporal association is weak, whereas local latent neighborhoods are temporally enriched relative to subtype-matched random baselines."
- "Supports SDE modeling" -> "motivates future reduced dynamical modeling, subject to forecasting validation."

## Claims That Can Be Strengthened

- The full-data cache is large and clearly auditable.
- Molecular Hamming correlations are robust across the tested pair-sampling seeds.
- HA+NA Hamming is more strongly associated with latent distance than HA-only Hamming in both subtypes in the current full-data results.
- Exact HA+NA duplicates are common and were explicitly removed for the local analyses.
- Local temporal-neighbor enrichment is large in magnitude after exact HA+NA deduplication.
- PCA and TwoNN agree qualitatively that the embedding cloud is much lower-dimensional than the nominal 384-dimensional representation.

## Bottom Line

The paper can become submission-ready as a careful geometric audit. It should not try to become a forecasting or antigenic-prediction manuscript without new evidence. The fastest path to a publishable candidate is:

1. rewrite the manuscript around audit logic and honest limitations;
2. verify and expand citations;
3. add at least one or two lightweight controls if time permits;
4. explicitly reserve antigenic, phylogenetic, and prospective claims for future work.
