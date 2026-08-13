# GISAID EpiFlu Clade Metadata Coverage Audit
Private metadata files were concatenated locally from the five GISAID EpiFlu metadata-only XLS exports generated from EPI_SET_260506bu. No sequences are redistributed here.
## Inputs
- `gisaid_epiflu_isolates20002014.xls`
- `gisaid_epiflu_isolates20142017.xls`
- `gisaid_epiflu_isolates20172019.xls`
- `gisaid_epiflu_isolates20192021.xls`
- `gisaid_epiflu_isolates20212022.xls`

Combined private CSV: `data/gisaid_metadata_private/gisaid_epiflu_isolates_2000_2022_epi_set_260506bu_combined.csv`
Joined private CSV: `data/gisaid_metadata_private/gisaid_epiflu_isolates_2000_2022_epi_set_260506bu_joined_dedup_cache.csv`

## Join Coverage Against Deduplicated HA+NA Cache
| quantity | count | fraction |
|---|---:|---:|
| Deduplicated cache records | 82,306 | 100.00% |
| Metadata unique Isolate_Id rows | 81,943 | 99.56% |
| Matched by epi_isl / Isolate_Id | 81,943 | 99.56% |
| Missing after join | 363 | 0.44% |
| Raw non-empty Clade values, including `unassigned` | 81,943 | 99.56% |
| Assigned Clade values, excluding `unassigned` | 76,238 | 92.63% |
| Assigned Lineage values | 31,305 | 38.03% |
| Assigned Genotype values | 0 | 0.00% |

`unassigned` is reported separately and is not treated as a biological clade label in the enrichment analysis.

## Coverage by Subtype
| subtype | cache n | matched | matched % | raw clade incl. unassigned | assigned clade excl. unassigned | lineage assigned | genotype assigned | raw `unassigned` clade |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H1N1 | 36,753 | 36,723 | 99.92% | 36,723 (99.92%) | 36,156 (98.38%) | 31,304 (85.17%) | 0 (0.00%) | 567 |
| H3N2 | 45,553 | 45,220 | 99.27% | 45,220 (99.27%) | 40,082 (87.99%) | 1 (0.00%) | 0 (0.00%) | 5,138 |

## Assigned Clade Coverage by Year and Subtype
| subtype | year | cache n | matched | assigned clade | assigned clade % | lineage assigned | lineage % |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1N1 | 2000 | 87 | 87 | 61 | 70.11% | 71 | 81.61% |
| H1N1 | 2001 | 113 | 113 | 95 | 84.07% | 107 | 94.69% |
| H1N1 | 2002 | 11 | 11 | 9 | 81.82% | 8 | 72.73% |
| H1N1 | 2003 | 30 | 30 | 20 | 66.67% | 15 | 50.00% |
| H1N1 | 2004 | 8 | 8 | 4 | 50.00% | 4 | 50.00% |
| H1N1 | 2005 | 55 | 55 | 40 | 72.73% | 43 | 78.18% |
| H1N1 | 2006 | 93 | 93 | 28 | 30.11% | 64 | 68.82% |
| H1N1 | 2007 | 476 | 476 | 373 | 78.36% | 414 | 86.97% |
| H1N1 | 2008 | 341 | 341 | 243 | 71.26% | 253 | 74.19% |
| H1N1 | 2009 | 4,761 | 4,761 | 4,627 | 97.19% | 3,380 | 70.99% |
| H1N1 | 2010 | 1,433 | 1,433 | 1,423 | 99.30% | 769 | 53.66% |
| H1N1 | 2011 | 1,066 | 1,066 | 1,054 | 98.87% | 604 | 56.66% |
| H1N1 | 2012 | 588 | 588 | 584 | 99.32% | 399 | 67.86% |
| H1N1 | 2013 | 1,353 | 1,353 | 1,343 | 99.26% | 1,010 | 74.65% |
| H1N1 | 2014 | 1,433 | 1,422 | 1,417 | 98.88% | 1,120 | 78.16% |
| H1N1 | 2015 | 1,908 | 1,908 | 1,898 | 99.48% | 1,618 | 84.80% |
| H1N1 | 2016 | 3,708 | 3,708 | 3,705 | 99.92% | 3,443 | 92.85% |
| H1N1 | 2017 | 2,053 | 2,034 | 2,031 | 98.93% | 1,963 | 95.62% |
| H1N1 | 2018 | 5,551 | 5,551 | 5,539 | 99.78% | 5,136 | 92.52% |
| H1N1 | 2019 | 7,228 | 7,228 | 7,223 | 99.93% | 6,477 | 89.61% |
| H1N1 | 2020 | 3,669 | 3,669 | 3,660 | 99.75% | 3,630 | 98.94% |
| H1N1 | 2021 | 526 | 526 | 518 | 98.48% | 516 | 98.10% |
| H1N1 | 2022 | 262 | 262 | 261 | 99.62% | 260 | 99.24% |
| H3N2 | 2000 | 170 | 161 | 0 | 0.00% | 0 | 0.00% |
| H3N2 | 2001 | 92 | 91 | 0 | 0.00% | 0 | 0.00% |
| H3N2 | 2002 | 211 | 200 | 0 | 0.00% | 0 | 0.00% |
| H3N2 | 2003 | 393 | 384 | 0 | 0.00% | 0 | 0.00% |
| H3N2 | 2004 | 334 | 321 | 0 | 0.00% | 0 | 0.00% |
| H3N2 | 2005 | 293 | 279 | 1 | 0.34% | 0 | 0.00% |
| H3N2 | 2006 | 131 | 129 | 0 | 0.00% | 0 | 0.00% |
| H3N2 | 2007 | 310 | 303 | 0 | 0.00% | 1 | 0.32% |
| H3N2 | 2008 | 352 | 326 | 0 | 0.00% | 0 | 0.00% |
| H3N2 | 2009 | 644 | 621 | 1 | 0.16% | 0 | 0.00% |
| H3N2 | 2010 | 679 | 668 | 3 | 0.44% | 0 | 0.00% |
| H3N2 | 2011 | 1,015 | 986 | 220 | 21.67% | 0 | 0.00% |
| H3N2 | 2012 | 1,506 | 1,443 | 739 | 49.07% | 0 | 0.00% |
| H3N2 | 2013 | 1,409 | 1,355 | 1,284 | 91.13% | 0 | 0.00% |
| H3N2 | 2014 | 2,243 | 2,196 | 2,191 | 97.68% | 0 | 0.00% |
| H3N2 | 2015 | 3,033 | 3,033 | 3,029 | 99.87% | 0 | 0.00% |
| H3N2 | 2016 | 3,841 | 3,841 | 3,834 | 99.82% | 0 | 0.00% |
| H3N2 | 2017 | 7,762 | 7,748 | 7,713 | 99.37% | 0 | 0.00% |
| H3N2 | 2018 | 5,105 | 5,105 | 5,097 | 99.84% | 0 | 0.00% |
| H3N2 | 2019 | 9,309 | 9,309 | 9,258 | 99.45% | 0 | 0.00% |
| H3N2 | 2020 | 1,555 | 1,555 | 1,552 | 99.81% | 0 | 0.00% |
| H3N2 | 2021 | 3,273 | 3,273 | 3,268 | 99.85% | 0 | 0.00% |
| H3N2 | 2022 | 1,893 | 1,893 | 1,892 | 99.95% | 0 | 0.00% |

## Most Frequent Assigned Clade Labels

### H1N1
| clade | count |
|---|---:|
| `6B.1` | 16,292 |
| `6B.1A.5a` | 3,965 |
| `6B.1A` | 3,133 |
| `6B.1A.5a.1` | 2,899 |
| `6B.1A.1` | 2,179 |
| `6B.1A.6` | 1,939 |
| `6B.1A.5b` | 1,643 |
| `6B.1A.5a.2` | 1,437 |
| `6B.1A.7` | 1,041 |
| `6B.2` | 469 |
| `6B.1A.2` | 438 |
| `6B.1A.3` | 264 |
| `6B.1A.5a.2a` | 230 |
| `6B.1A.5` | 127 |
| `6B.1A.5a.2a.1` | 97 |

### H3N2
| clade | count |
|---|---:|
| `3C.2a2` | 4,936 |
| `3C.3a1` | 4,214 |
| `3C.2a` | 4,064 |
| `3C.2a1` | 3,791 |
| `3C.2a1b.1` | 3,251 |
| `3C.2a3` | 2,005 |
| `3C.2a1b.2b` | 1,995 |
| `3C.2a1b.2` | 1,780 |
| `3C.2a1b.2a.2a.3` | 1,727 |
| `3C.3` | 1,707 |
| `3C.2` | 1,565 |
| `3C.3a` | 1,559 |
| `3C.2a1b.1b` | 1,154 |
| `3C.2a1b.2a.2a.1a` | 1,113 |
| `3C` | 858 |

## Methodological Reading
- The join via `epi_isl`/`Isolate_Id` is excellent: 99.56% of deduplicated cache records were recovered in the GISAID metadata export.
- Assigned clade coverage remains high after treating `unassigned` as missing: 92.63% globally, 98.38% for H1N1, and 88.00% for H3N2.
- `Lineage` is useful for H1N1 but essentially absent for H3N2 in this export, so clade labels are the appropriate metadata field for the cross-subtype enrichment analysis.
- The clade analysis should be described as clade-label enrichment / evolutionary-taxonomic coherence, not as full phylogenetic validation or antigenic validation.
