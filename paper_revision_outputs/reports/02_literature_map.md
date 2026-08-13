# Literature Map

Date: 2026-05-05

Scope: references needed to turn the current thesis-checkpoint draft into a journal-style manuscript on the geometry of AntigenLM latent representations for Influenza A HA/NA records.

## Verification Notes

I verified the key bibliographic facts from primary publisher pages, arXiv/OpenReview records, university repositories, or journal pages where available. I did not add speculative or unverifiable references to `references.bib`.

Key sources consulted include:

- AntigenLM arXiv record: `https://arxiv.org/abs/2602.09067`
- AntigenLM OpenReview/ICLR PDF: `https://openreview.net/pdf?id=Y0zPlHDO5p`
- DNABERT Bioinformatics page: `https://academic.oup.com/bioinformatics/article/37/15/2112/6128680`
- Nucleotide Transformer Nature Methods page: `https://www.nature.com/articles/s41592-024-02523-z`
- Hie et al. Science entry via Princeton: `https://collaborate.princeton.edu/en/publications/learning-the-language-of-viral-evolution-and-escape/`
- Smith et al. antigenic cartography record: `https://repub.eur.nl/pub/58381`
- Koel et al. antigenic drift record: `https://repub.eur.nl/pub/63769`
- Bedford et al. eLife article page/PMC record: `https://pmc.ncbi.nlm.nih.gov/articles/PMC3909918/`
- Facco et al. TwoNN paper: DOI `10.1038/s41598-017-11873-y`
- Levina and Bickel NeurIPS page: `https://papers.neurips.cc/paper/2577-maximum-likelihood-estimation-of-intrinsic-dimension`
- DANCo record: DOI `10.1016/j.patcog.2014.02.013`
- Luksza and Lassig Nature page: `https://www.nature.com/articles/nature13087`
- Neher et al. eLife page: `https://elifesciences.org/articles/03568/figures`
- Shu and McCauley GISAID citation: DOI `10.2807/1560-7917.ES.2017.22.13.30494`
- Distance correlation paper: DOI `10.1214/009053607000000505`
- Lee and Verleysen dimensionality-reduction quality criteria: DOI `10.1016/j.neucom.2008.12.017`

## Cluster 1: AntigenLM and Exact Model Context

### Added

- `pei2026antigenlm`

### Why Needed

The manuscript audits representations derived from a local AntigenLM-like prediction checkpoint. It must cite the model paper and distinguish this local audit from the AntigenLM authors' forecasting and subtype-classification claims.

### Where to Cite

- Introduction: motivation for auditing AntigenLM rather than a generic sequence model.
- Methods: model/checkpoint description.
- Discussion/Limitations: explicit non-reproduction of AntigenLM forecasting.

### Important Framing

AntigenLM is now verifiable as accepted by ICLR 2026 via arXiv/OpenReview. The local repository still needs to clarify whether the checkpoint is an official released checkpoint, a locally reconstructed checkpoint, or a task-specific model matching the AntigenLM architecture. The manuscript should cite the AntigenLM paper but should not imply official reproduction unless proven.

## Cluster 2: DNA and Protein Language Models

### Added

- `ji2021dnabert`
- `dallatorre2025nucleotide`
- `rives2021biological`
- `hie2021viral`

### Why Needed

The introduction needs to position AntigenLM within biological sequence language modeling. DNABERT and Nucleotide Transformer are canonical DNA/genomic representation examples. Rives et al. is a canonical protein language model paper. Hie et al. is especially relevant because it connects viral protein language modeling with influenza HA, viral escape, and immune recognition.

### Where to Cite

- Introduction: biological sequence models and learned representations.
- Discussion: what sequence-model embeddings can and cannot imply biologically.

### Optional Additional References

- `nguyen2023hyenadna` was added as a verified arXiv/NeurIPS-style reference for long-context genomic modeling, but it is optional for the revised paper. It is most useful if the manuscript mentions long-context genomic models.

## Cluster 3: Influenza Evolution, Antigenic Cartography, and Molecular Versus Antigenic Distance

### Added

- `smith2004mapping`
- `koel2013substitutions`
- `bedford2014integrating`

### Why Needed

The paper must make a clean distinction between sequence similarity, phylogenetic/evolutionary relatedness, and antigenic similarity. Smith et al. established antigenic cartography for influenza. Koel et al. shows that specific HA substitutions near the receptor-binding site can drive major antigenic change. Bedford et al. integrates antigenic dynamics with molecular evolution and is useful for explaining why antigenic validation requires serological or antigenic data, not Hamming distance alone.

### Where to Cite

- Introduction: influenza HA/NA evolution and antigenic drift.
- Discussion/Limitations: molecular Hamming distance is not antigenic distance.
- Future work: antigenic cartography, HI assay distances, phylogenetic validation.

## Cluster 4: Influenza Forecasting, Phylodynamics, and Surveillance

### Added

- `luksza2014fitness`
- `neher2014predicting`
- `hadfield2018nextstrain`
- `shu2017gisaid`

### Why Needed

The manuscript explicitly does not perform forecasting. Still, it should acknowledge the established forecasting and phylodynamic context so that the embedding audit reads as a preparatory validation step, not as a substitute for established evolutionary prediction.

### Where to Cite

- Introduction: why state-space or representation audits matter before downstream evolutionary models.
- Discussion: what would be needed for forecasting validation.
- Data availability/reproducibility: GISAID citation and access restrictions.

## Cluster 5: PCA and Effective Dimension

### Already Present / Retained

- `jolliffe2002pca`
- `jolliffe2016pca`

### Why Needed

The paper uses PCA cumulative variance thresholds and participation ratio as linear effective-dimension diagnostics. These references are appropriate for PCA background.

### Where to Cite

- Methods: PCA computation and interpretation.
- Results: PCA effective dimension.
- Discussion: PCA dimension is not intrinsic manifold dimension.

### Missing or Optional

No separate participation-ratio citation was added. The manuscript can define the participation ratio directly and treat it as an eigenvalue-summary statistic. If the paper makes a stronger theoretical claim about participation ratio, add a dedicated reference manually after verification.

## Cluster 6: Intrinsic Dimension and Manifold Diagnostics

### Added / Retained

- `facco2017intrinsic`
- `levina2005maximum`
- `ceruti2014danco`

### Why Needed

The current paper uses TwoNN. Reviewers will expect limitations and alternatives. Levina-Bickel MLE and DANCo are useful canonical alternatives to mention as future robustness checks.

### Where to Cite

- Methods: TwoNN.
- Discussion/Limitations: limitations of the estimator and alternative estimators.
- Future work: complementary intrinsic-dimension estimators.

## Cluster 7: Metric Validation and Neighborhood Preservation

### Added

- `spearman1904`
- `mantel1967detection`
- `szekely2007distance`
- `lee2009quality`

### Why Needed

Spearman is used directly. Mantel tests, distance correlation, and rank/neighborhood preservation metrics are not currently run, but they are appropriate future or supplementary diagnostics for validating the relationship between latent distances and molecular/temporal distances.

### Where to Cite

- Methods: Spearman correlation.
- Discussion/Limitations: sampled pair correlations are only one metric-validity diagnostic.
- Future work: Mantel-style tests, distance correlation, trustworthiness/continuity, kNN retrieval.

### Caution

Mantel-style tests must be described carefully because pairwise distance entries are not independent and permutation assumptions can be fragile in structured evolutionary data. If the manuscript cites Mantel tests, it should frame them as a supplementary diagnostic, not as a decisive validation.

## Missing References Not Added Yet

These may be useful, but I did not add them because the current manuscript does not require them or because they need more targeted verification before use:

- BLOSUM or substitution-matrix references for protein-aware distance metrics.
- Codon-aware or dN/dS distance references.
- Current WHO influenza vaccine-strain selection methodology references.
- Nextstrain clade-label documentation beyond the main Nextstrain paper.
- beth-1 or other recent influenza deep-learning forecasting papers if the discussion expands into direct forecasting comparisons.
- Specific GISAID data-access and acknowledgement instructions for the exact downloaded dataset; these often need project-specific acknowledgement text rather than only a citation.
- scikit-learn implementation references, if software citation requirements are added.

## Risky or Unverifiable References

- The original bibliography entry for `pei2026antigenlm` said "metadata should be verified." That risk is now partly resolved: arXiv confirms submission on 2026-02-09, DOI `10.48550/arXiv.2602.09067`, and acceptance by ICLR 2026; OpenReview PDF shows the published conference-paper version. The local checkpoint provenance remains a separate repository-level issue.
- Older exploratory thesis outputs cite dynamic/SDE results from `figures/gisaid/` and `results/master_results_summary.md`; those should not be imported into this paper unless the manuscript is expanded beyond the latent-geometry audit.
- Any statement that "molecular similarity implies antigenic similarity" is scientifically risky and should not be made without antigenic data.

## Recommended Citation Placement in the Revised Manuscript

- Abstract: no citations needed.
- Introduction:
  - AntigenLM and genomic sequence models: `pei2026antigenlm`, `ji2021dnabert`, `dallatorre2025nucleotide`, `rives2021biological`, `hie2021viral`.
  - Influenza evolution and antigenic drift: `smith2004mapping`, `koel2013substitutions`, `bedford2014integrating`.
  - Forecasting context: `luksza2014fitness`, `neher2014predicting`, `hadfield2018nextstrain`.
- Methods:
  - Data source: `shu2017gisaid`.
  - Spearman: `spearman1904`.
  - PCA: `jolliffe2002pca`, `jolliffe2016pca`.
  - TwoNN: `facco2017intrinsic`.
- Discussion/Limitations:
  - Antigenic data limitations: `smith2004mapping`, `bedford2014integrating`, `koel2013substitutions`.
  - Alternative intrinsic-dimension methods: `levina2005maximum`, `ceruti2014danco`.
  - Alternative metric diagnostics: `mantel1967detection`, `szekely2007distance`, `lee2009quality`.

## Bibliography Update

Updated file:

- `papers/paper_1_latent_geometry_full/references.bib`

The update replaces the provisional AntigenLM entry and adds verified entries with DOI, venue, URL/arXiv metadata where available.
