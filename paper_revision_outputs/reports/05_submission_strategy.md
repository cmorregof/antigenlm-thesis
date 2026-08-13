# Submission Strategy

Date: 2026-05-05

## Realistic Status

Assessment: **submit after minor analysis** for BMC Bioinformatics or Scientific Reports if the manuscript remains framed as a careful geometric audit. **Submit after major analysis** for Virus Evolution, Bioinformatics, PLOS Computational Biology, or Algorithms for Molecular Biology.

The revised paper is much closer to a publishable candidate than the original checkpoint draft, but it still needs at least one robustness control to avoid looking like a descriptive internal thesis artifact.

## Recommended Target Journal Ranking

1. **BMC Bioinformatics**
2. **Scientific Reports**
3. **Algorithms for Molecular Biology**
4. **Virus Evolution**
5. **Bioinformatics**
6. **PLOS Computational Biology**

## Best-Fit Framing by Journal

### 1. BMC Bioinformatics

Best framing:

> A reproducible computational audit of latent geometry in a biological sequence language model, using Influenza A HA/NA embeddings as a high-value viral case study.

Why it fits:

- BMC Bioinformatics can accept transparent computational analyses when the methods are clear, reproducible, and useful to other researchers.
- The current manuscript's main contribution is methodological scrutiny rather than a new biological discovery.

Required before submission:

- Add software versions and exact commands.
- Add at least one negative control, preferably date permutation for temporal-neighborhood enrichment.
- Add deduplicated Spearman correlations or explicitly justify why they are deferred.
- Strengthen data availability/GISAID access language.

Status:

- Submit after minor analysis.

### 2. Scientific Reports

Best framing:

> A large-scale descriptive study showing that learned influenza sequence representations are molecularly organized, locally temporally coherent, and low-dimensional under complementary diagnostics.

Why it fits:

- The scope is broad and empirical.
- The paper can succeed if it is technically sound and the claims are conservative.

Required before submission:

- Add at least one robustness control.
- Make limitations prominent.
- Keep all claims descriptive and avoid implying biological mechanism.

Status:

- Submit after minor analysis, but reviewer variance may be high.

### 3. Algorithms for Molecular Biology

Best framing:

> A general audit framework for biological sequence-model embeddings, demonstrated on AntigenLM and influenza HA/NA.

Why it fits:

- The journal fit improves if the paper emphasizes algorithmic diagnostics: metric validity, intrinsic dimension, temporal-neighborhood enrichment, and controls.

Required before submission:

- Recast the manuscript as a reusable framework.
- Add pseudocode or a formal workflow.
- Add at least one baseline model or simulated/random embedding control.
- Make scripts clean enough to be reusable by others.

Status:

- Submit after major analysis/reframing.

### 4. Virus Evolution

Best framing:

> Learned latent neighborhoods of influenza HA/NA sequences show molecular organization and local temporal coherence, motivating phylogenetically informed representation audits.

Why it is currently weaker:

- Virus Evolution reviewers will expect phylogenetic, clade, lineage, antigenic, or phylodynamic validation.
- The current paper has no tree, no clade labels, and no antigenic data.

Required before submission:

- Add clade or lineage validation.
- Add phylogenetic distance or tree-neighborhood comparison.
- Stratify temporal-neighborhood results by clade/year if labels are available.
- Discuss H1N1 and H3N2 biology more deeply.

Status:

- Submit after major analysis.

### 5. Bioinformatics

Best framing:

> A reusable software/benchmark suite for auditing learned biological sequence representations.

Why it is currently weak:

- The current manuscript is a case study, not yet a method paper or benchmark.
- Bioinformatics will likely expect broad applicability, software maturity, comparative baselines, or a novel algorithm.

Required before submission:

- Package the audit pipeline.
- Add at least two model or embedding baselines.
- Evaluate across more datasets or representations.
- Present a clear reusable contribution beyond AntigenLM.

Status:

- Submit after major analysis and software reframing.

### 6. PLOS Computational Biology

Best framing:

> What learned viral representations reveal about molecular similarity, evolutionary time, and candidate state spaces.

Why it is currently weak:

- PLOS Computational Biology typically requires stronger biological insight or mechanistic/computational novelty.
- The current evidence is an audit, not a biological advance.

Required before submission:

- Add antigenic, phylogenetic, or prospective validation.
- Demonstrate a biological insight not obvious from sequence similarity alone.
- Compare to established evolutionary models or phylodynamic baselines.

Status:

- Not yet; submit after major analysis.

## Where the Current Paper Is Closest

Closest current fit:

1. BMC Bioinformatics
2. Scientific Reports

Moderate fit after reframing:

3. Algorithms for Molecular Biology

Needs substantial additional biological validation:

4. Virus Evolution

Not currently close:

5. Bioinformatics
6. PLOS Computational Biology

## Required Changes Before Each Target

| Journal | Current closeness | Required changes |
|---|---|---|
| BMC Bioinformatics | High | Add robustness controls, software versions, data availability details |
| Scientific Reports | Medium-high | Add one negative control, sharpen biological caveats |
| Algorithms for Molecular Biology | Medium | Reframe as algorithm/workflow, add baseline/control diagnostics |
| Virus Evolution | Medium-low | Add phylogenetic, clade, lineage, or antigenic validation |
| Bioinformatics | Low | Package reusable method/benchmark and compare against baselines |
| PLOS Computational Biology | Low | Add substantial biological insight and external validation |

## Originally Recommended Next Experiment Before Submission

The pre-robustness recommendation was to run a compact robustness panel:

1. **Date-permutation control for local temporal coherence**
   - Shuffle year/month labels within subtype after deduplication.
   - Recompute nearest-neighbor temporal enrichment.
   - Expected interpretation: if true labels show much smaller neighbor time differences than permuted labels, local temporal coherence is less likely to be a generic density artifact.

2. **Deduplicated pairwise Spearman correlations**
   - Repeat current molecular and temporal Spearman analyses after exact HA+NA deduplication.
   - This addresses the concern that duplicate/near-duplicate records inflate molecular correlations.

3. **NA-only and amino-acid HA/NA distance analyses**
   - Add NA-only nucleotide Hamming.
   - Translate coding sequences where safe and compute amino-acid Hamming.
   - This improves biological interpretability and clarifies whether HA+NA signal is HA-dominated.

Minimum publishable addition:

- Date-permutation control plus deduplicated Spearman.

Best cost-benefit addition:

- Date permutation, deduplicated Spearman, and NA-only distance.

Status: completed in `paper_revision_outputs/robustness_panel_summary.md` and folded into the manuscript.

## Submit Now?

Recommendation after the robustness panel: **close to submit after manuscript polish and reproducibility cleanup** for BMC Bioinformatics or Scientific Reports; **not yet** for Virus Evolution, Bioinformatics, or PLOS Computational Biology.

The previous blocker "submitting without a control panel" has been partly addressed by the focused robustness panel in `paper_revision_outputs/robustness_panel_summary.md`. The paper now includes exact HA+NA-deduplicated Spearman correlations, NA-only nucleotide correlations, simple amino-acid Hamming proxy checks, and a date-label permutation control for local temporal neighborhoods. The remaining high-priority cleanup is reproducibility wording, GISAID/data-availability language, checkpoint provenance, and local LaTeX compilation in an environment with TeX installed.

## Final Target Recommendation

Primary target: **BMC Bioinformatics**.

Rationale:

- The paper's strongest contribution is a transparent computational audit of learned biological sequence representations.
- The current evidence supports a methods-aware data-analysis framing.
- The journal fit does not require proving antigenic phenotype, forecasting skill, or a new algorithm, provided the methodology is reproducible and claims remain conservative.

Backup target: **Scientific Reports**.

Rationale:

- Broad scope and tolerance for descriptive empirical studies.
- Requires careful wording and robust controls to avoid being viewed as a narrow case study.

## Post-Robustness Status

Minimum publishable additions completed:

- Date-permutation control.
- Deduplicated Spearman correlations.
- NA-only molecular-distance analysis.

Additional useful but nonessential additions before BMC Bioinformatics:

- Software-version table or supplement.
- Exact command log in the manuscript supplement.
- Journal-compliant GISAID acknowledgement/access statement.
- Clarification of local checkpoint provenance.
- Optional kNN molecular retrieval precision or local rank-preservation diagnostics.

Current recommendation:

- **BMC Bioinformatics:** submit after minor editorial/reproducibility cleanup.
- **Scientific Reports:** submit after minor editorial/reproducibility cleanup, with broad-scope framing.
- **Virus Evolution:** still requires major evolutionary validation.
- **Bioinformatics:** still requires method/package/benchmark reframing.
- **PLOS Computational Biology:** still requires substantially stronger biological insight.
