# Abstract draft

Large language models trained on viral sequence data are increasingly used as
representation spaces for evolutionary analysis, yet their geometry is rarely
audited before being used for downstream dynamical modeling. We present a
geometric audit of the AntigenLM latent space for Influenza A H1N1 and H3N2
using cached embeddings from paired HA and NA records. We compare Euclidean
latent distances with temporal distance and with molecular distances based on
normalized Hamming distance in HA and HA+NA. In the main stratified sample,
global temporal correlation is weak, whereas molecular correlations are strong:
latent distance correlates substantially with Hamming HA and HA+NA distances.
We further examine effective and intrinsic dimension using PCA, participation
ratio, and TwoNN after exact HA+NA deduplication. Both spectral and nearest-neighbor
analyses suggest a low-dimensional, strongly anisotropic representation. Finally,
we evaluate local temporal neighborhoods and find that nearest neighbors in
latent space are separated by only a few months, much less than random pairs
from the same subtype; this signal persists after removing exact HA+NA
duplicates. These results suggest that AntigenLM embeddings capture biologically
meaningful molecular structure and locally coherent temporal organization, rather
than merely encoding chronological order. However, this audit does not demonstrate
successful sequence prediction, latent decoding, or the validity of any specific
stochastic differential equation. Instead, it provides a methodological foundation
and a set of diagnostic criteria for deciding whether continuous latent dynamics
are scientifically defensible.

