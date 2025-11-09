//! # kalnal
//!
//! `kalnal` is a k-mer based contig clustering tool that uses spatial distribution
//! of k-mers (interval histograms) to infer phylogenetic relationships between contigs.
//!
//! The tool implements Neighbor-Joining tree construction with bootstrap support
//! to cluster contigs, which can be useful for separating subgenomes in polyploid assemblies.
