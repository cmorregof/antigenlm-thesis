# Alignment-Aware Robustness Panel

This panel uses Biopython global pairwise alignments on a modest sampled set of deduplicated within-subtype pairs. It is a robustness check, not a curated multiple sequence alignment or protein-level antigenic analysis.

| distance proxy | segment | subtype | rho mean | rho sd | valid pairs mean |
|---|---|---|---:|---:|---:|
| prefix | HA | H1N1 | 0.8203 | 0.0347 | 250 |
| prefix | HA | H3N2 | 0.5714 | 0.0721 | 250 |
| prefix | NA | H1N1 | 0.7644 | 0.0235 | 250 |
| prefix | NA | H3N2 | 0.5264 | 0.0544 | 249 |
| prefix | HA+NA | H1N1 | 0.8369 | 0.0248 | 250 |
| prefix | HA+NA | H3N2 | 0.6272 | 0.0633 | 249 |
| aligned | HA | H1N1 | 0.6435 | 0.0342 | 250 |
| aligned | HA | H3N2 | 0.5927 | 0.0366 | 250 |
| aligned | NA | H1N1 | 0.4639 | 0.0572 | 250 |
| aligned | NA | H3N2 | 0.5646 | 0.0258 | 250 |
| aligned | HA+NA | H1N1 | 0.5869 | 0.0449 | 250 |
| aligned | HA+NA | H3N2 | 0.6159 | 0.0268 | 250 |

Interpretation: persistence of positive correlations under the aligned proxy supports that the scalable shared-prefix Hamming result is not solely a prefix-length artifact. The small sample and simple global alignment keep this as a limited robustness panel.
