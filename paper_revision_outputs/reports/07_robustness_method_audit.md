# Robustness Panel Method Audit

Date: 2026-05-05

Audited files:

- `paper_revision_outputs/run_robustness_panel.py`
- `paper_revision_outputs/robustness_panel_results.json`
- `paper_revision_outputs/robustness_panel_summary.md`
- `paper_revision_outputs/robustness_panel_run.log`
- `paper_revision_outputs/06_robustness_panel_update.md`
- `papers/paper_1_latent_geometry_full/main.tex`

No new experiments were run for this audit. I inspected the script, generated outputs, run log, update note, and manuscript tables/claims.

## 1. Exact Input Files Used

The robustness run used:

- Primary data/cache input: `results/embeddings_cache_full_all_available.pkl`
- Robustness script: `paper_revision_outputs/run_robustness_panel.py`
- Imported helper module: `latent_geometry_full_analysis.py`

The script imported these helper functions/constants from `latent_geometry_full_analysis.py`:

- `SUBTYPE_ORDER`
- `deduplicate_by_ha_na`
- `load_cache`
- `month_index`
- `normalized_hamming_arrays`
- `sample_pairs`
- `stats`

The script did **not** load AntigenLM, did **not** regenerate embeddings, did **not** read raw FASTA files directly, and did **not** use `results/latent_geometry_full_metrics.json` as an input for its calculations.

Generated robustness outputs:

- `paper_revision_outputs/robustness_panel_results.json`
- `paper_revision_outputs/robustness_panel_summary.md`
- `paper_revision_outputs/robustness_panel_run.log`

## 2. Records Before and After Exact HA+NA Deduplication

Before deduplication:

- Total records/embeddings: `111,756`
- H1N1: `46,125`
- H3N2: `65,631`

After exact HA+NA deduplication:

- Total records/embeddings: `82,306`
- H1N1: `36,753`
- H3N2: `45,553`

Removed exact HA+NA duplicates:

- H1N1: `9,372`
- H3N2: `20,078`
- Total removed: `29,450`

These counts are traceable to `robustness_panel_results.json`, which reports `deduplicated_n = 82306` and `removed_duplicates = {"H3N2": 20078, "H1N1": 9372}` for both the deduplicated Spearman and temporal-permutation components.

## 3. Deduplication Method

Deduplication used `deduplicate_by_ha_na` from `latent_geometry_full_analysis.py`.

Mechanism:

- For each record in cache order, a key was computed as SHA1 of `ha_sequence|na_sequence`.
- If the key had not been seen, the record was retained.
- If the key had already been seen, the record was excluded and counted as a duplicate for its subtype.
- The first representative of each exact HA+NA sequence pair was retained.

This is exact sequence-pair deduplication. It does not remove near-duplicates, synonymous duplicates at amino-acid level, records with only HA or only NA duplication, or sequences differing by ambiguous characters.

## 4. Subtype Separation

Yes. H1N1 and H3N2 were analyzed separately throughout the robustness panel.

Evidence:

- Pair sampling in `deduplicated_spearman` loops over `SUBTYPE_ORDER` and sets `idx = np.where(td == subtype)[0]` before sampling pairs.
- Temporal permutation in `temporal_permutation_control` also loops over `SUBTYPE_ORDER` and uses only indices for the current subtype.
- Output rows in `robustness_panel_results.json` are subtype-specific.

No cross-subtype pairwise distances or mixed-subtype nearest-neighbor graphs were used in the robustness panel.

## 5. Random Pairs Sampled

The script sampled `200,000` non-self random pairs per subtype per seed.

For each seed and subtype:

- H1N1: `200,000` requested pairs
- H3N2: `200,000` requested pairs

With three seeds, this yields `600,000` requested pairs per subtype for each metric, before metric-specific omissions due to length-tolerance failures.

## 6. Random Seeds

Pair-sampled Spearman seeds:

- `42`
- `7`
- `123`

Temporal date-permutation seeds:

- `42`
- `7`
- `123`

These are reported in `robustness_panel_results.json` under `parameters.pair_seeds` and `parameters.temporal_permutation_seeds`.

## 7. Bootstrap Confidence Intervals

No bootstrap confidence intervals were computed.

The outputs report:

- seed means;
- seed standard deviations;
- mean valid-pair counts;
- mean omitted-pair counts.

For temporal permutation, the outputs report mean and standard deviation across the three permutation seeds for median time differences. These are not bootstrap confidence intervals.

## 8. Molecular Distance Computation

### Nucleotide HA

Metric: `nt_hamming_ha`

Method:

- Convert HA nucleotide string to uppercase ASCII byte array.
- Compare two HA arrays using `normalized_hamming_arrays`.
- If either sequence is empty, return `None`.
- If length difference exceeds 5% of the longer length, return `None`.
- Otherwise compare the shared prefix of length `min(len_a, len_b)`.
- Distance is mismatch count divided by the shorter length.

### Nucleotide NA

Metric: `nt_hamming_na`

Same as nucleotide HA, but applied to NA nucleotide arrays.

### Nucleotide HA+NA

Metric: `nt_hamming_ha_na`

Method:

- Compute nucleotide HA Hamming distance as above.
- Compute nucleotide NA Hamming distance as above.
- If either segment distance is `None`, omit the pair for HA+NA.
- Weight segment distances by the shorter segment lengths:

```text
(d_HA * min_HA_length + d_NA * min_NA_length) / (min_HA_length + min_NA_length)
```

### Amino-Acid HA+NA

Metric: `aa_hamming_ha_na`

Method:

- Translate HA nucleotide string to a frame-0 amino-acid string.
- Translate NA nucleotide string to a frame-0 amino-acid string.
- Unknown/ambiguous codons are mapped to `X`.
- Trailing nucleotides not forming a full codon are dropped.
- Convert translated amino-acid strings to ASCII byte arrays.
- Compute HA and NA amino-acid Hamming distances with the same `normalized_hamming_arrays` function and 5% length-tolerance rule.
- Combine HA and NA amino-acid distances using the same shorter-length weighted average.

The script also computed `aa_hamming_ha` and `aa_hamming_na` and included them in the JSON/summary, although the manuscript table only reports amino-acid HA+NA.

## 9. Sequence-Length Handling

Mechanically, sequence-length differences were handled consistently with the existing paper method:

- Pairs with length difference greater than 5% of the longer sequence were omitted for that segment.
- Pairs within the 5% tolerance were compared on the shared prefix only.
- Combined HA+NA distances were omitted if either HA or NA exceeded the tolerance.
- Combined distances were weighted by the shorter segment lengths.

This is acceptable for the stated nucleotide Hamming proxy and consistent with the prior full-data analysis. However, it is not a full biological alignment procedure. It does not perform codon-aware alignment, indel alignment, signal-peptide/HA numbering alignment, or curated HA/NA protein alignment. The amino-acid proxy inherits these limitations.

## 10. Amino-Acid Proxy Computation

The amino-acid proxy was computed by `translate_nt` in `run_robustness_panel.py`.

Exact behavior:

- Uppercase nucleotide string.
- Use frame 0 starting at the first nucleotide.
- Translate codons using a standard genetic-code dictionary.
- Map any codon not in the dictionary, including ambiguous codons containing non-ACGT symbols, to `X`.
- Keep stop codons as `*`.
- Drop trailing 1 or 2 nucleotides if sequence length is not divisible by 3.

This is a simple robustness proxy. It is not a curated protein sequence extraction, not a validated reading-frame correction, not a multiple sequence alignment, and not an antigenic-site-aware distance.

## 11. Date-Label Permutation Method

In `temporal_permutation_control`:

1. Exact HA+NA duplicates were removed first.
2. For each subtype separately, the script built a latent nearest-neighbor graph from the deduplicated embeddings.
3. It computed true neighbor temporal deltas using the observed month indices.
4. For each permutation seed, it permuted the vector of month indices within the same subtype using `rng.permutation(t)`.
5. It recomputed neighbor temporal deltas using the same neighbor indices but permuted month labels.
6. It also computed subtype-random pair deltas using random non-self pairs of the same count as the neighbor deltas.

The month index was computed as:

```text
year * 12 + clipped_month
```

where month is clipped to `[1, 12]` by the imported `month_index` helper.

## 12. Were Dates Permuted Within Subtype Only?

Yes.

The permutation was performed after selecting records for one subtype:

```python
idx = np.where(td == subtype)[0]
t = t_months_all[idx]
perm_t = rng.permutation(t)
```

Thus H1N1 dates were permuted only among H1N1 records, and H3N2 dates only among H3N2 records.

## 13. Were Embeddings Fixed While Only Dates Were Permuted?

Yes.

The neighbor graph was computed once from the original deduplicated embeddings:

```python
nn.fit(X)
_, neighbor_idx = nn.kneighbors(X)
neighbor_idx = neighbor_idx[:, 1:]
```

The same `neighbor_idx` was then reused for both true and permuted date labels. Embeddings and nearest-neighbor identities were held fixed; only date labels were permuted.

## 14. k Values Tested

The temporal nearest-neighbor panel tested:

- `k=5`
- `k=10`
- `k=20`

These are reported in `robustness_panel_results.json`, `robustness_panel_summary.md`, and `main.tex`.

## 15. Nearest-Neighbor Self-Matches

Yes, self-matches were excluded.

The script requested `max(k_values) + 1` neighbors and then removed the first neighbor:

```python
_, neighbor_idx = nn.kneighbors(X)
neighbor_idx = neighbor_idx[:, 1:]
```

Because the nearest neighbor returned by scikit-learn for points queried against their own fitted dataset is the point itself at distance zero, this removes self-matches before taking the top-k neighbors.

## 16. Exact Duplicates Before Nearest-Neighbor Analysis

Yes.

`temporal_permutation_control` calls `deduplicate_by_ha_na` before building subtype-specific nearest-neighbor graphs. Therefore exact HA+NA duplicates were excluded before temporal nearest-neighbor analysis.

## 17. Are New Manuscript Claims Supported?

Mostly yes. The new manuscript claims are traceable to `robustness_panel_results.json` and `robustness_panel_summary.md`.

Supported claims:

- "Exact HA+NA deduplication reduces but does not remove the molecular-distance signal."
  - Supported by deduplicated nucleotide HA+NA rho means: H1N1 `0.8358`, H3N2 `0.6315`.
- "NA-only nucleotide correlations are lower than HA-only correlations in both subtypes."
  - Supported by H1N1: NA `0.7677` vs HA `0.8086`; H3N2: NA `0.5287` vs HA `0.5883`.
- "Combined HA+NA nucleotide proxy remains strongest among nucleotide-distance summaries."
  - Supported by H1N1: HA+NA `0.8358` > HA `0.8086` > NA `0.7677`; H3N2: HA+NA `0.6315` > HA `0.5883` > NA `0.5287`.
- "Simple translated amino-acid HA+NA proxies remain substantially correlated with latent distance."
  - Supported by amino-acid HA+NA rho means: H1N1 `0.7563`, H3N2 `0.6131`.
- "Date-label permutation moves median neighbor time differences back to random-like values."
  - Supported by true medians `2.00` months versus permuted means `42.33-42.67` months in H1N1 and `35.00` months in H3N2; random medians match these values closely.
- "The latent neighbor graph is fixed and collection months are permuted within subtype."
  - Supported by script logic.

## 18. Claims Needing Qualification

The manuscript is mostly appropriately qualified, but the following points should remain explicit:

- "Robust" should mean robust to these specific controls only: exact HA+NA deduplication, NA-only molecular proxy, simple amino-acid proxy, and within-subtype date permutation.
- The amino-acid analysis is a proxy, not a curated protein alignment. It should not be used to claim protein-level antigenic validity.
- Date permutation controls for marginal date distribution within subtype, but not for geography, host, lab, clade, lineage, or sampling-burst structure.
- The temporal control does not prove causal evolutionary dynamics or forecasting skill.
- Seed standard deviations are not confidence intervals.
- Pairwise sampled distances are dependent, so p-values should remain deemphasized.

No new claim currently says antigenic similarity, vaccine relevance, phylogenetic validity, or prospective forecasting has been established; that restraint should be preserved.

## 19. Traceability of New Tables in `main.tex`

### Table `tab:robust-spearman`

The table values in `main.tex` are traceable to `robustness_panel_results.json` under:

```text
deduplicated_spearman.aggregate
```

Trace:

- Temporal H1N1: `0.0690`, `0.0031` from `temporal`, H1N1.
- Temporal H3N2: `0.1497`, `0.0017` from `temporal`, H3N2.
- Nucleotide HA H1N1: `0.8086`, `0.0013` from `nt_hamming_ha`, H1N1.
- Nucleotide HA H3N2: `0.5883`, `0.0006` from `nt_hamming_ha`, H3N2.
- Nucleotide NA H1N1: `0.7677`, `0.0010` from `nt_hamming_na`, H1N1.
- Nucleotide NA H3N2: `0.5287`, `0.0025` from `nt_hamming_na`, H3N2.
- Nucleotide HA+NA H1N1: `0.8358`, `0.0010` from `nt_hamming_ha_na`, H1N1.
- Nucleotide HA+NA H3N2: `0.6315`, `0.0015` from `nt_hamming_ha_na`, H3N2.
- Amino-acid HA+NA H1N1: `0.7563`, `0.0013` from `aa_hamming_ha_na`, H1N1.
- Amino-acid HA+NA H3N2: `0.6131`, `0.0008` from `aa_hamming_ha_na`, H3N2.

### Table `tab:temporal-permutation`

The table values in `main.tex` are traceable to `robustness_panel_results.json` under:

```text
temporal_permutation.aggregate
```

Trace:

- H1N1 k=5: true `2.00`, permuted `42.33`, ratio `0.047`.
- H1N1 k=10: true `2.00`, permuted `42.67`, ratio `0.047`.
- H1N1 k=20: true `2.00`, permuted `42.67`, ratio `0.047`.
- H3N2 k=5: true `2.00`, permuted `35.00`, ratio `0.057`.
- H3N2 k=10: true `2.00`, permuted `35.00`, ratio `0.057`.
- H3N2 k=20: true `2.00`, permuted `35.00`, ratio `0.057`.

The manuscript table omits permutation standard deviations and random medians, but the caption and surrounding text accurately describe the control. If space permits, adding random medians would improve transparency, but the current values are traceable.

## 20. Remaining Methodological Risks

- No bootstrap confidence intervals were computed.
- Pairwise sampled distances are dependent; seed standard deviations reflect sampling-seed variability, not independent uncertainty.
- Hamming distance uses shared-prefix comparison under a 5% length-tolerance rule, not biological sequence alignment.
- The amino-acid proxy uses frame-0 translation without verifying coding frame, mature protein boundaries, segment annotations, stop codons, or alignment.
- Ambiguous codons become `X`, which may affect amino-acid proxy distances.
- Exact HA+NA deduplication does not remove near-duplicates or biologically equivalent amino-acid duplicates.
- Date permutation is within subtype and preserves the latent neighbor graph, which is appropriate for the stated negative control, but it does not control for geography, host, laboratory, clade, lineage, or sampling bursts.
- The subtype-random and date-permutation controls do not establish phylogenetic validity.
- No antigenic assay, HI, neutralization, or antigenic-cartography validation is present.
- No kNN molecular retrieval precision, trustworthiness/continuity, distance correlation, Mantel-style test, or local rank-preservation analysis was added.
- No prospective forecasting validation was added.
- No random-embedding, matched-spectrum random-embedding, PCA-whitened, or untrained-checkpoint control was added.

## Bottom Line

The robustness panel is methodologically traceable and supports the new manuscript tables and conservative claims. Its strongest additions are exact HA+NA-deduplicated molecular correlations and a within-subtype date-label permutation control with fixed latent neighbor graphs. The main qualifications are that amino-acid distances are simple proxy translations, seed summaries are not bootstrap confidence intervals, and the controls still do not validate antigenic, phylogenetic, or forecasting claims.
