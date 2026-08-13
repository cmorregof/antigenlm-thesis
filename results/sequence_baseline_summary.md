# Sequence Baseline Neighborhood Results

This analysis writes aggregate results only. It does not redistribute sequences, isolate identifiers, accessions, or accession-level metadata.

## Parameters

- query sample per subtype: `2000` assigned-clade records
- k values: `[5, 10, 20]`
- nucleotide k-mer size: `5`
- Hamming candidate pool: `500` k-mer neighbors
- random projection dimension: `384`

## Clade Precision and Temporal Neighborhoods

| method | subtype | k | queries | precision@k | valid neighbors | median months | IQR months | p10-p90 months | <=6 mo |
|---|---|---:|---:|---:|---:|---:|---|---|---:|
| AntigenLM latent | H1N1 | 5 | 2,000 | 0.9037 | 4.97 | 2.0 | 1.0--5.0 | 0.0--12.0 | 0.815 |
| AntigenLM latent | H1N1 | 10 | 2,000 | 0.8808 | 9.92 | 2.0 | 1.0--5.0 | 0.0--13.0 | 0.783 |
| AntigenLM latent | H1N1 | 20 | 2,000 | 0.8525 | 19.84 | 2.0 | 1.0--7.0 | 0.0--15.0 | 0.746 |
| k-mer TF-IDF | H1N1 | 5 | 2,000 | 0.9857 | 4.99 | 1.0 | 0.0--3.0 | 0.0--5.0 | 0.931 |
| k-mer TF-IDF | H1N1 | 10 | 2,000 | 0.9829 | 9.97 | 1.0 | 1.0--3.0 | 0.0--6.0 | 0.916 |
| k-mer TF-IDF | H1N1 | 20 | 2,000 | 0.9787 | 19.92 | 2.0 | 1.0--3.0 | 0.0--7.0 | 0.895 |
| candidate-retrieved Hamming | H1N1 | 5 | 2,000 | 0.9847 | 4.98 | 1.0 | 1.0--3.0 | 0.0--6.0 | 0.918 |
| candidate-retrieved Hamming | H1N1 | 10 | 2,000 | 0.9785 | 9.95 | 1.0 | 1.0--3.0 | 0.0--6.0 | 0.902 |
| candidate-retrieved Hamming | H1N1 | 20 | 2,000 | 0.9739 | 19.91 | 2.0 | 1.0--4.0 | 0.0--7.0 | 0.878 |
| k-mer random projection | H1N1 | 5 | 2,000 | 0.9859 | 4.99 | 1.0 | 0.0--3.0 | 0.0--5.0 | 0.932 |
| k-mer random projection | H1N1 | 10 | 2,000 | 0.9829 | 9.96 | 1.0 | 1.0--3.0 | 0.0--6.0 | 0.917 |
| k-mer random projection | H1N1 | 20 | 2,000 | 0.9786 | 19.91 | 2.0 | 1.0--3.0 | 0.0--7.0 | 0.895 |
| subtype random | H1N1 | 5 | 2,000 | 0.2414 | 5.00 | 40.0 | 14.0--85.0 | 4.0--115.0 | 0.133 |
| subtype random | H1N1 | 10 | 2,000 | 0.2399 | 10.00 | 41.0 | 14.0--84.0 | 4.0--115.0 | 0.135 |
| subtype random | H1N1 | 20 | 2,000 | 0.2405 | 20.00 | 40.0 | 14.0--84.0 | 4.0--115.0 | 0.133 |
| AntigenLM latent | H3N2 | 5 | 2,000 | 0.8683 | 4.97 | 1.0 | 1.0--4.0 | 0.0--10.0 | 0.841 |
| AntigenLM latent | H3N2 | 10 | 2,000 | 0.8394 | 9.92 | 2.0 | 1.0--4.0 | 0.0--11.0 | 0.814 |
| AntigenLM latent | H3N2 | 20 | 2,000 | 0.8008 | 19.82 | 2.0 | 1.0--5.0 | 0.0--13.0 | 0.783 |
| k-mer TF-IDF | H3N2 | 5 | 2,000 | 0.9661 | 4.99 | 1.0 | 0.0--3.0 | 0.0--5.0 | 0.929 |
| k-mer TF-IDF | H3N2 | 10 | 2,000 | 0.9581 | 9.99 | 1.0 | 0.0--3.0 | 0.0--6.0 | 0.911 |
| k-mer TF-IDF | H3N2 | 20 | 2,000 | 0.9483 | 19.96 | 1.0 | 1.0--3.0 | 0.0--7.0 | 0.892 |
| candidate-retrieved Hamming | H3N2 | 5 | 2,000 | 0.9638 | 4.99 | 1.0 | 0.0--3.0 | 0.0--6.0 | 0.919 |
| candidate-retrieved Hamming | H3N2 | 10 | 2,000 | 0.9559 | 9.98 | 1.0 | 1.0--3.0 | 0.0--7.0 | 0.899 |
| candidate-retrieved Hamming | H3N2 | 20 | 2,000 | 0.9431 | 19.97 | 2.0 | 1.0--3.0 | 0.0--7.0 | 0.877 |
| k-mer random projection | H3N2 | 5 | 2,000 | 0.9654 | 4.99 | 1.0 | 0.0--3.0 | 0.0--5.0 | 0.929 |
| k-mer random projection | H3N2 | 10 | 2,000 | 0.9580 | 9.99 | 1.0 | 0.0--3.0 | 0.0--6.0 | 0.912 |
| k-mer random projection | H3N2 | 20 | 2,000 | 0.9480 | 19.95 | 1.0 | 1.0--3.0 | 0.0--7.0 | 0.892 |
| subtype random | H3N2 | 5 | 2,000 | 0.0690 | 5.00 | 26.0 | 12.0--47.0 | 3.0--65.0 | 0.153 |
| subtype random | H3N2 | 10 | 2,000 | 0.0680 | 10.00 | 27.0 | 12.0--47.0 | 3.0--64.0 | 0.155 |
| subtype random | H3N2 | 20 | 2,000 | 0.0701 | 20.00 | 27.0 | 12.0--47.0 | 3.0--64.0 | 0.153 |

## Interpretation Guardrails

- If AntigenLM matches or trails Hamming/k-mer neighborhoods, the correct interpretation is that it recapitulates substantial sequence-derived structure in a learned compressed representation.
- If AntigenLM exceeds these baselines, the improvement is only under these local clade/temporal metrics and is not antigenic, phenotypic, or forecasting validation.
- Candidate-retrieved Hamming is not an exhaustive all-vs-all Hamming kNN baseline; it is a practical approximation reranking a k-mer candidate set.
