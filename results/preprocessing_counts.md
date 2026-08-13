# Preprocessing and Cache Construction Counts

This file contains aggregate counts only. It intentionally omits sequences, accessions, isolate names, locations, hosts, submitters, and accession-level metadata.

## Main Count Table

| Step | Retained records | Removed at step | Notes |
|---|---:|---:|---|
| Raw local GISAID/EpiFlu records loaded | not retained in current logs |  | Private raw-export count not reconstructed. |
| Processed paired records in local JSON | 111,756 |  | Starting point for this reproducible audit. |
| Records with HA, NA, subtype, year, and month | 111,756 | 0 from processed JSON | All processed paired records satisfy these fields. |
| Successfully embedded records | 111,756 | 0 from processed paired set | No missing cache entries among processed paired records. |
| Records in final cache | 111,756 |  | Mean-pooled 384-dimensional vectors. |
| Exact HA+NA deduplicated records | 82,306 | 29,450 | Exact duplicate HA+NA strings collapsed for local-neighborhood analyses. |

## Counts by Subtype

| Subtype | processed paired | cache records | exact HA+NA deduplicated | HA median length | NA median length | records with non-ACGT/IUPAC-or-gap ambiguity |
|---|---:|---:|---:|---:|---:|---:|
| H1N1 | 46,125 | 46,125 | 36,753 | 1734 | 1420 | 528 |
| H3N2 | 65,631 | 65,631 | 45,553 | 1735 | 1436 | 1,869 |

## Scope Notes

- The reproducible counts begin at the processed `paired_strains` JSON files.
- Counts for earlier raw GISAID parsing/filtering stages are not retained in the current logs.
- The processed JSON records contain `epi_isl`, subtype, year/month/day, HA sequence, and NA sequence, but no host, geography, laboratory, or passage filters are applied in this audit script.
- Host, geography, passage, and submitter information are available only in private metadata exports and are not redistributed in these outputs.
