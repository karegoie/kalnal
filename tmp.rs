use kodama::{linkage, Method};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::error::Error;

/// Represents a dendrogram node from hierarchical clustering
#[derive(Debug, Clone)]
pub struct Dendrogram {
    pub linkage: Vec<(usize, usize, f64, usize)>,
}

/// Performs hierarchical clustering using Ward's method
///
/// # Arguments
/// * `data` - 2D array where each row is a sample and columns are features (kmer counts)
///
/// # Returns
/// * A Dendrogram structure containing linkage information
pub fn hierarchical_clustering(data: Array2<f64>) -> Result<Dendrogram, Box<dyn Error>> {
    let n_samples = data.nrows();
    
    // Convert ndarray to condensed distance matrix format for kodama
    let condensed = compute_distance_matrix(&data);
    
    // Perform hierarchical clustering with Ward's method
    let dendrogram = linkage(&mut condensed.to_vec(), n_samples, Method::Ward);
    
    // Convert kodama::Dendrogram to our format
    let linkage_info: Vec<(usize, usize, f64, usize)> = dendrogram
        .steps()
        .iter()
        .map(|step| {
            (
                step.cluster1,
                step.cluster2,
                step.dissimilarity,
                step.size,
            )
        })
        .collect();
    
    Ok(Dendrogram {
        linkage: linkage_info,
    })
}

/// Compute pairwise Jaccard distances between samples
fn compute_distance_matrix(data: &Array2<f64>) -> Array1<f64> {
    let n = data.nrows();
    let n_distances = (n * (n - 1)) / 2;
    let mut distances = Array1::zeros(n_distances);
    
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let row_i = data.row(i);
            let row_j = data.row(j);
            
            // Jaccard distance = 1 - Jaccard Index
            // Jaccard Index = |A ∩ B| / |A ∪ B|
            // For kmer counts, we treat non-zero values as presence
            let mut intersection = 0.0;
            let mut union = 0.0;
            
            for (a, b) in row_i.iter().zip(row_j.iter()) {
                let a_present = *a > 0.0;
                let b_present = *b > 0.0;
                
                if a_present && b_present {
                    intersection += 1.0;
                }
                if a_present || b_present {
                    union += 1.0;
                }
            }
            
            let jaccard_index = if union > 0.0 {
                intersection / union
            } else {
                0.0 // Both samples have no kmers
            };
            
            let dist = 1.0 - jaccard_index;
            distances[idx] = dist;
            idx += 1;
        }
    }
    
    distances
}

/// Convert kmer count HashMap to a matrix format suitable for clustering
///
/// # Arguments
/// * `kmer_counts` - HashMap where keys are sample names and values are HashMaps of kmer->count
///
/// # Returns
/// * Tuple of (data matrix, sample names)
pub fn prepare_data_matrix(
    kmer_counts: &HashMap<String, HashMap<String, i32>>,
) -> Result<(Array2<f64>, Vec<String>), Box<dyn Error>> {
    let n_samples = kmer_counts.len();
    
    if n_samples == 0 {
        return Err("No samples provided".into());
    }
    
    // Collect all unique kmers across all samples
    let mut all_kmers: Vec<String> = kmer_counts
        .values()
        .flat_map(|counts| counts.keys().cloned())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    
    all_kmers.sort(); // Ensure consistent ordering
    
    let n_features = all_kmers.len();
    let mut data = Array2::zeros((n_samples, n_features));
    let mut sample_names = Vec::new();
    
    // Fill the matrix
    for (i, (sample_name, sample_counts)) in kmer_counts.iter().enumerate() {
        sample_names.push(sample_name.clone());
        
        for (j, kmer) in all_kmers.iter().enumerate() {
            let count = sample_counts.get(kmer).copied().unwrap_or(0);
            data[[i, j]] = count as f64;
        }
    }
    
    Ok((data, sample_names))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_distance_matrix() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
        let distances = compute_distance_matrix(&data);
        
        // Distance between sample 0 {1,0} and sample 1 {1,1}
        // Intersection: 1 (first kmer), Union: 2
        // Jaccard Index = 1/2 = 0.5, Distance = 1 - 0.5 = 0.5
        assert!((distances[0] - 0.5).abs() < 1e-10);
        
        // Distance between sample 0 {1,0} and sample 2 {0,1}
        // Intersection: 0, Union: 2
        // Jaccard Index = 0/2 = 0.0, Distance = 1.0
        assert!((distances[1] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_prepare_data_matrix() {
        let mut kmer_counts = HashMap::new();
        
        let mut sample1 = HashMap::new();
        sample1.insert("AAA".to_string(), 5);
        sample1.insert("AAT".to_string(), 3);
        kmer_counts.insert("sample1".to_string(), sample1);
        
        let mut sample2 = HashMap::new();
        sample2.insert("AAA".to_string(), 2);
        sample2.insert("AAT".to_string(), 7);
        kmer_counts.insert("sample2".to_string(), sample2);
        
        let (data, names) = prepare_data_matrix(&kmer_counts).unwrap();
        
        assert_eq!(data.nrows(), 2);
        assert_eq!(data.ncols(), 2);
        assert_eq!(names.len(), 2);
    }
}
use std::{env, error::Error, path::PathBuf};

/// Parsing command line k-size and fasta file path arguments
pub struct Config {
    pub k: usize,
    pub fasta_path: PathBuf,
}

impl Config {
    pub fn new(mut args: env::Args) -> Result<Config, Box<dyn Error>> {
        let k: usize = match args.nth(1) {
            Some(arg) => match arg.parse() {
                Ok(k) if k > 0 && k < 33 => k,
                Ok(_) => return Err("k-mer length needs to be larger than zero and no more than 32".into()),
                Err(_) => return Err(format!("issue with k-mer length argument: {}", arg).into()),
            },
            None => return Err("k-mer length input required".into()),
        };

        let fasta_path = match args.next() {
            Some(arg) => PathBuf::from(arg),
            None => return Err("fasta file path argument needed".into()),
        };

        Ok(Config { k, fasta_path })
    }
}
custom_error::custom_error! { pub ValidityError
    InvalidByte = "not a valid byte",
}

/// A valid k-mer
#[derive(Debug, Eq, PartialEq)]
pub struct Kmer(pub Vec<u8>);

impl Kmer {
    fn new() -> Self {
        Self(Vec::new())
    }

    fn add(&mut self, elem: Monomer) {
        self.0.push(elem.into_u8())
    }

    pub(crate) fn from_sub(sub: &[u8]) -> Result<Self, ValidityError> {
        sub.iter().map(|b| Monomer::try_from(*b)).collect()
    }

    pub(crate) fn canonical(
        reverse_complement: Vec<u8>,
        kmer: Vec<u8>,
    ) -> impl Iterator<Item = u8> {
        match reverse_complement.cmp(&kmer) {
            std::cmp::Ordering::Less => reverse_complement.into_iter(),
            _ => kmer.into_iter(),
        }
    }

    pub(crate) fn find_invalid(sub: &[u8]) -> usize {
        sub.iter()
            .rposition(|byte| Monomer::try_from(*byte).is_err())
            .unwrap()
    }
}

impl FromIterator<u8> for Kmer {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        let mut k = Kmer::new();

        for i in iter {
            k.add(Monomer::from_u8(i));
        }
        k
    }
}

pub(crate) enum Monomer {
    A(u8),
    C(u8),
    G(u8),
    T(u8),
}

impl TryFrom<u8> for Monomer {
    type Error = ValidityError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            b'A' => Ok(Self::A(value)),
            b'C' => Ok(Self::C(value)),
            b'G' => Ok(Self::G(value)),
            b'T' => Ok(Self::T(value)),
            _ => Err(ValidityError::InvalidByte),
        }
    }
}

impl FromIterator<Monomer> for Kmer {
    fn from_iter<I: IntoIterator<Item = Monomer>>(iter: I) -> Self {
        let mut k = Self::new();

        for i in iter {
            k.add(i);
        }
        k
    }
}

#[allow(clippy::from_over_into)]
impl Into<u64> for Monomer {
    fn into(self) -> u64 {
        match self {
            Self::A(_) => 0,
            Self::C(_) => 1,
            Self::G(_) => 2,
            Self::T(_) => 3,
        }
    }
}

impl Monomer {
    fn complement(self) -> Self {
        match self {
            Self::A(_) => Self::from_u8(b'T'),
            Self::C(_) => Self::from_u8(b'G'),
            Self::G(_) => Self::from_u8(b'C'),
            Self::T(_) => Self::from_u8(b'A'),
        }
    }

    fn from_u8(byte: u8) -> Self {
        match byte {
            b'A' => Self::A(byte),
            b'C' => Self::C(byte),
            b'G' => Self::G(byte),
            _ => Self::T(byte),
        }
    }

    fn into_u8(self) -> u8 {
        match self {
            Self::A(b) => b,
            Self::C(b) => b,
            Self::G(b) => b,
            Self::T(b) => b,
        }
    }

    fn unpack(bit: u64) -> Self {
        match bit {
            0 => Self::from_u8(b'A'),
            1 => Self::from_u8(b'C'),
            2 => Self::from_u8(b'G'),
            _ => Self::from_u8(b'T'),
        }
    }
}

trait Pack {
    fn isolate(self, i: usize, k: usize) -> Self;
    fn replace(self) -> Self;
}

impl Pack for u64 {
    fn isolate(self, i: usize, k: usize) -> Self {
        self << ((i * 2) + 64 - (k * 2))
    }

    fn replace(self) -> Self {
        self >> 62
    }
}

/// Compressing k-mers of length `0 < k < 33`, bitpacking them into unsigned integers
pub(crate) struct Bitpack(pub u64);

impl Bitpack {
    fn new() -> Self {
        Self(0)
    }

    fn pack(&mut self, elem: u8) {
        self.shift();
        let mask: u64 = Monomer::from_u8(elem).into();
        self.0 |= mask
    }

    fn shift(&mut self) {
        self.0 <<= 2
    }
}

impl FromIterator<u8> for Bitpack {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        let mut c = Self::new();

        for i in iter {
            c.pack(i)
        }
        c
    }
}

/// Unpack bitpacked k-mer data
#[derive(Hash, PartialEq, Eq)]
pub struct Unpack(pub Vec<u8>);

impl Unpack {
    fn new() -> Self {
        Unpack(Vec::new())
    }

    fn add(&mut self, elem: u8) {
        self.0.push(elem);
    }

    pub(crate) fn bit(bit: u64, k: usize) -> Self {
        (0..k)
            .into_iter()
            .map(|i| bit.isolate(i, k))
            .map(|bit| bit.replace())
            .map(Monomer::unpack)
            .map(|m| m.into_u8())
            .collect()
    }
}

impl FromIterator<u8> for Unpack {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        let mut c = Unpack::new();

        for i in iter {
            c.add(i)
        }
        c
    }
}

/// Convert a DNA string slice into its [reverse compliment](https://en.wikipedia.org/wiki/Complementarity_(molecular_biology)#DNA_and_RNA_base_pair_complementarity).
pub(crate) struct RevComp(pub Vec<u8>);

impl FromIterator<u8> for RevComp {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        let mut c = RevComp::new();

        for i in iter {
            c.add(i)
        }
        c
    }
}

impl RevComp {
    fn new() -> Self {
        Self(vec![])
    }

    fn add(&mut self, elem: u8) {
        self.0.push(elem)
    }

    pub(crate) fn from_kmer(kmer: &Kmer) -> Self {
        kmer.0
            .iter()
            .rev()
            .map(|byte| Monomer::from_u8(*byte).complement().into_u8())
            .collect()
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

    #[test]
    fn test_from_valid_substring() {
        let sub = &[b'G', b'A', b'T', b'T', b'A', b'C', b'A'];
        let k = Kmer::from_sub(sub).unwrap();
        insta::assert_snapshot!(format!("{:?}", k), @"Kmer([71, 65, 84, 84, 65, 67, 65])");
    }

    #[test]
    fn test_parse_valid_byte() {
        let b = b'N';
        assert!(Monomer::try_from(b).is_err());
    }

    #[test]
    fn from_substring_returns_err_for_invalid_substring() {
        let sub = &[b'N'];
        let k = Kmer::from_sub(sub);
        assert!(k.is_err());
    }

    #[test]
    fn find_invalid_works() {
        let dna = "NACNN".as_bytes();
        let ans = Kmer::find_invalid(dna);
        assert_eq!(4, ans);
        assert_eq!(&b'N', dna.iter().collect::<Vec<_>>()[ans]);

        let dna = "NACNG".as_bytes();
        let ans = Kmer::find_invalid(dna);
        assert_eq!(3, ans);
        assert_eq!(&b'N', dna.iter().collect::<Vec<_>>()[ans]);

        let dna = "NANTG".as_bytes();
        let ans = Kmer::find_invalid(dna);
        assert_eq!(2, ans);
        assert_eq!(&b'N', dna.iter().collect::<Vec<_>>()[ans]);

        let dna = "NNCTG".as_bytes();
        let ans = Kmer::find_invalid(dna);
        assert_eq!(1, ans);
        assert_eq!(&b'N', dna.iter().collect::<Vec<_>>()[ans]);

        let dna = "NACTG".as_bytes();
        let ans = Kmer::find_invalid(dna);
        assert_eq!(0, ans);
        assert_eq!(&b'N', dna.iter().collect::<Vec<_>>()[ans]);
    }
}
//! # kalnal
//!
//! `kalnal` is a k-mer based contig clustering tool that uses spatial distribution
//! of k-mers (interval histograms) to infer phylogenetic relationships between contigs.
//!
//! The tool implements Neighbor-Joining tree construction with bootstrap support
//! to cluster contigs, which can be useful for separating subgenomes in polyploid assemblies.
use bio::io::fasta;
use clap::Parser;
use itertools::Itertools;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::Write;
use std::process;

/// CLI tool for clustering contigs based on k-mer spatial distribution
#[derive(Parser, Debug)]
#[command(name = "kalnal")]
#[command(about = "Cluster contigs using k-mer interval histograms with bootstrap support", long_about = None)]
struct Args {
    /// K-mer length
    k: usize,

    /// Input FASTA file path
    fasta_file: String,

    /// Output Newick file path
    output_file: String,

    /// Number of k-mers to sample (also used as bootstrap replicates)
    #[arg(short = 'n', long = "n-kmers", default_value_t = 1000)]
    n_kmers: usize,
}

// Log-scale histogram bins (powers of 2)
const BINS: &[usize] = &[0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, usize::MAX];

fn main() {
    let args = Args::parse();

    if let Err(e) = run(args) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading contigs from {}...", args.fasta_file);
    
    // Data Loading
    let reader = fasta::Reader::from_file(&args.fasta_file)?;
    let mut contigs: Vec<(String, Vec<u8>)> = Vec::new();
    let mut contig_ids: Vec<String> = Vec::new();

    for result in reader.records() {
        let record = result?;
        let id = record.id().to_string();
        let seq = record.seq().to_vec();
        contig_ids.push(id.clone());
        contigs.push((id, seq));
    }

    if contigs.is_empty() {
        return Err("No contigs found in input file".into());
    }

    eprintln!("Loaded {} contigs", contigs.len());
    eprintln!("Finding unique k-mers (k={})...", args.k);

    // Find all unique k-mers
    let all_kmers = find_unique_kmers(&contigs, args.k);
    eprintln!("Found {} unique k-mers", all_kmers.len());

    if all_kmers.is_empty() {
        return Err("No valid k-mers found".into());
    }

    // Determine how many k-mers to sample
    let n_sample = args.n_kmers.min(all_kmers.len());
    eprintln!("Randomly sampling {} k-mers...", n_sample);

    // Randomly select k-mers
    let mut rng = thread_rng();
    let selected_kmers: Vec<Vec<u8>> = all_kmers
        .choose_multiple(&mut rng, n_sample)
        .cloned()
        .collect();

    eprintln!("Building tree for each k-mer and collecting bipartitions...");
    
    // Build a tree for EACH k-mer and collect all bipartitions
    let all_bipartitions: Vec<BTreeSet<BTreeSet<String>>> = selected_kmers
        .par_iter()
        .enumerate()
        .map(|(i, kmer)| {
            if i % 100 == 0 && i > 0 {
                eprintln!("  Processed {}/{} k-mers", i, n_sample);
            }
            
            // Calculate histogram for this k-mer
            let histogram = calculate_interval_histogram(kmer, &contigs, BINS);
            
            // Build distance matrix from this single k-mer's histogram
            let dist_matrix = histogram_to_distance_matrix(&histogram, BINS.len() - 1);
            
            // Build tree from distance matrix
            let tree_newick = build_nj_tree_simple(&dist_matrix, &contig_ids);
            
            // Parse and extract bipartitions
            parse_bipartitions_from_newick(&tree_newick, &contig_ids)
        })
        .collect();

    eprintln!("Processed {} trees from k-mers", all_bipartitions.len());

    // Bootstrap: resample k-mer trees with replacement
    eprintln!("Running {} bootstrap replicates...", args.n_kmers);
    
    let bootstrap_bipartitions: Vec<BTreeSet<BTreeSet<String>>> = (0..args.n_kmers)
        .into_par_iter()
        .map(|i| {
            if i % 10 == 0 && i > 0 {
                eprintln!("  Bootstrap {}/{}", i, args.n_kmers);
            }
            
            // Resample trees (k-mer trees) with replacement
            let mut rng = thread_rng();
            let resampled_indices: Vec<usize> = (0..n_sample)
                .map(|_| rng.gen_range(0..all_bipartitions.len()))
                .collect();
            
            // Collect all bipartitions from resampled trees
            let mut combined_bipartitions = BTreeSet::new();
            for idx in resampled_indices {
                for bipartition in &all_bipartitions[idx] {
                    combined_bipartitions.insert(bipartition.clone());
                }
            }
            
            combined_bipartitions
        })
        .collect();

    // Aggregate bootstrap counts
    let mut bootstrap_counts: HashMap<BTreeSet<String>, usize> = HashMap::new();
    for bipartitions in bootstrap_bipartitions {
        for bipartition in bipartitions {
            *bootstrap_counts.entry(bipartition).or_insert(0) += 1;
        }
    }

    eprintln!("Building consensus tree from all k-mer trees...");
    
    // Build consensus tree: use the first k-mer's tree structure and annotate with support
    let first_histogram = calculate_interval_histogram(&selected_kmers[0], &contigs, BINS);
    let first_dist_matrix = histogram_to_distance_matrix(&first_histogram, BINS.len() - 1);
    let consensus_newick = build_nj_tree_simple(&first_dist_matrix, &contig_ids);
    
    // Annotate with bootstrap support
    let annotated_newick = annotate_newick_with_support(
        &consensus_newick,
        &bootstrap_counts,
        args.n_kmers,
        &contig_ids,
    );

    let mut output = File::create(&args.output_file)?;
    writeln!(output, "{}", annotated_newick)?;

    eprintln!("Success! Tree written to {}", args.output_file);
    Ok(())
}

/// Find all unique canonical k-mers in the dataset (excluding k-mers with 'N')
fn find_unique_kmers(contigs: &[(String, Vec<u8>)], k: usize) -> Vec<Vec<u8>> {
    let mut kmer_set: BTreeSet<Vec<u8>> = BTreeSet::new();

    for (_id, seq) in contigs {
        for window in seq.windows(k) {
            // Skip k-mers containing N
            if window.iter().any(|&b| b == b'N' || b == b'n') {
                continue;
            }
            let canonical = canonical_kmer(window);
            kmer_set.insert(canonical);
        }
    }

    kmer_set.into_iter().collect()
}

/// Get canonical k-mer (lexicographically smaller of k-mer and reverse complement)
fn canonical_kmer(kmer: &[u8]) -> Vec<u8> {
    let rc = reverse_complement(kmer);
    if rc < kmer.to_vec() {
        rc
    } else {
        kmer.to_vec()
    }
}

/// Calculate reverse complement of a DNA sequence
fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&b| match b {
            b'A' | b'a' => b'T',
            b'T' | b't' => b'A',
            b'C' | b'c' => b'G',
            b'G' | b'g' => b'C',
            _ => b'N',
        })
        .collect()
}

/// Calculate interval histogram for a k-mer across all contigs
fn calculate_interval_histogram(
    kmer: &[u8],
    contigs: &[(String, Vec<u8>)],
    bins: &[usize],
) -> Array1<f64> {
    let n_bins = bins.len() - 1;
    let n_contigs = contigs.len();
    let mut histogram = Array1::<f64>::zeros(n_contigs * n_bins);

    for (contig_idx, (_id, seq)) in contigs.iter().enumerate() {
        // Find all positions of this k-mer in this contig
        let positions: Vec<usize> = seq
            .windows(kmer.len())
            .enumerate()
            .filter(|(_pos, window)| {
                let window_canonical = canonical_kmer(window);
                window_canonical == kmer
            })
            .map(|(pos, _)| pos)
            .collect();

        // Calculate intervals between consecutive positions
        let intervals: Vec<usize> = positions
            .iter()
            .tuple_windows()
            .map(|(a, b)| b - a)
            .collect();

        // Bin the intervals
        let mut contig_hist = vec![0.0; n_bins];
        for interval in intervals {
            for bin_idx in 0..n_bins {
                if interval >= bins[bin_idx] && interval < bins[bin_idx + 1] {
                    contig_hist[bin_idx] += 1.0;
                    break;
                }
            }
        }

        // Place into the full histogram
        for bin_idx in 0..n_bins {
            histogram[contig_idx * n_bins + bin_idx] = contig_hist[bin_idx];
        }
    }

    histogram
}

/// Convert histogram to distance matrix between contigs
fn histogram_to_distance_matrix(histogram: &Array1<f64>, n_bins: usize) -> Array2<f64> {
    let n_contigs = histogram.len() / n_bins;
    let mut dist_matrix = Array2::<f64>::zeros((n_contigs, n_contigs));
    
    for i in 0..n_contigs {
        for j in (i + 1)..n_contigs {
            let mut dist = 0.0;
            for bin_idx in 0..n_bins {
                let val_i = histogram[i * n_bins + bin_idx];
                let val_j = histogram[j * n_bins + bin_idx];
                dist += (val_i - val_j).powi(2);
            }
            dist = dist.sqrt();
            dist_matrix[[i, j]] = dist;
            dist_matrix[[j, i]] = dist;
        }
    }
    
    dist_matrix
}

/// Build NJ tree from distance matrix using simple UPGMA
fn build_nj_tree_simple(dist_matrix: &Array2<f64>, labels: &[String]) -> String {
    let n = dist_matrix.nrows();
    
    if n < 2 {
        return format!("{};", labels[0]);
    }
    
    if n == 2 {
        let dist = dist_matrix[[0, 1]];
        return format!("({}:{},{}:{});", labels[0], dist / 2.0, labels[1], dist / 2.0);
    }
    
    // Simple UPGMA clustering
    let mut clusters: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
    let mut distances = dist_matrix.clone();
    let mut active: Vec<bool> = vec![true; n];
    
    while clusters.iter().filter(|_| true).count() > 1 {
        // Find minimum distance pair
        let mut min_dist = f64::INFINITY;
        let mut min_i = 0;
        let mut min_j = 1;
        
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                if distances[[i, j]] < min_dist {
                    min_dist = distances[[i, j]];
                    min_i = i;
                    min_j = j;
                }
            }
        }
        
        // Merge clusters
        let new_cluster = format!("({}:{},{}:{})", 
            clusters[min_i], min_dist / 2.0,
            clusters[min_j], min_dist / 2.0);
        
        // Update distance matrix (UPGMA: average distance)
        for k in 0..n {
            if !active[k] || k == min_i || k == min_j {
                continue;
            }
            let new_dist = (distances[[min_i, k]] + distances[[min_j, k]]) / 2.0;
            distances[[min_i, k]] = new_dist;
            distances[[k, min_i]] = new_dist;
        }
        
        clusters[min_i] = new_cluster;
        active[min_j] = false;
        
        // Check if only one cluster remains
        let active_count = active.iter().filter(|&&a| a).count();
        if active_count == 1 {
            break;
        }
    }
    
    // Find the last active cluster
    for i in 0..n {
        if active[i] {
            return format!("{};", clusters[i]);
        }
    }
    
    // Fallback
    format!("({});", labels.join(","))
}

/// Parse bipartitions from Newick string
fn parse_bipartitions_from_newick(newick: &str, all_labels: &[String]) -> BTreeSet<BTreeSet<String>> {
    let mut bipartitions = BTreeSet::new();
    
    // Simple parser: find all groups in parentheses
    let mut stack: Vec<BTreeSet<String>> = Vec::new();
    let mut current_label = String::new();
    
    for ch in newick.chars() {
        match ch {
            '(' => {
                stack.push(BTreeSet::new());
            }
            ',' => {
                if !current_label.is_empty() {
                    let label = clean_label(&current_label);
                    if !label.is_empty() {
                        if let Some(set) = stack.last_mut() {
                            set.insert(label);
                        }
                    }
                    current_label.clear();
                }
            }
            ')' => {
                if !current_label.is_empty() {
                    let label = clean_label(&current_label);
                    if !label.is_empty() {
                        if let Some(set) = stack.last_mut() {
                            set.insert(label);
                        }
                    }
                    current_label.clear();
                }
                
                if let Some(clade) = stack.pop() {
                    if !clade.is_empty() && clade.len() < all_labels.len() {
                        bipartitions.insert(clade.clone());
                    }
                    
                    // Add to parent
                    if let Some(parent) = stack.last_mut() {
                        parent.extend(clade);
                    }
                }
            }
            ':' | ';' => {
                if !current_label.is_empty() {
                    let label = clean_label(&current_label);
                    if !label.is_empty() {
                        if let Some(set) = stack.last_mut() {
                            set.insert(label);
                        }
                    }
                    current_label.clear();
                }
                // Skip until next important character
            }
            _ => {
                if ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
                    current_label.push(ch);
                }
            }
        }
    }
    
    bipartitions
}

fn clean_label(label: &str) -> String {
    label.split(':').next().unwrap_or(label).trim().to_string()
}

/// Annotate Newick string with bootstrap support values
fn annotate_newick_with_support(
    newick: &str,
    bootstrap_counts: &HashMap<BTreeSet<String>, usize>,
    n_replicates: usize,
    all_labels: &[String],
) -> String {
    // For simplicity, we'll parse the tree and add support values at internal nodes
    // This is a simplified version - you may want to use a proper tree library
    
    let mut result = String::new();
    let mut _depth = 0;
    let mut current_clade: Vec<BTreeSet<String>> = Vec::new();
    let mut label_buffer = String::new();
    
    for ch in newick.chars() {
        match ch {
            '(' => {
                result.push(ch);
                _depth += 1;
                current_clade.push(BTreeSet::new());
            }
            ')' => {
                if !label_buffer.is_empty() {
                    if let Some(set) = current_clade.last_mut() {
                        set.insert(clean_label(&label_buffer));
                    }
                    label_buffer.clear();
                }
                
                result.push(ch);
                
                if let Some(clade) = current_clade.pop() {
                    if !clade.is_empty() && clade.len() > 1 && clade.len() < all_labels.len() {
                        let count = bootstrap_counts.get(&clade).copied().unwrap_or(0);
                        let support = (count as f64 * 100.0 / n_replicates as f64).round() as usize;
                        result.push_str(&format!("{}", support));
                    }
                    
                    if let Some(parent) = current_clade.last_mut() {
                        parent.extend(clade);
                    }
                }
                
                _depth -= 1;
            }
            ',' => {
                if !label_buffer.is_empty() {
                    if let Some(set) = current_clade.last_mut() {
                        set.insert(clean_label(&label_buffer));
                    }
                    label_buffer.clear();
                }
                result.push(ch);
            }
            ':' => {
                if !label_buffer.is_empty() {
                    result.push_str(&label_buffer);
                    if let Some(set) = current_clade.last_mut() {
                        set.insert(clean_label(&label_buffer));
                    }
                    label_buffer.clear();
                }
                result.push(ch);
            }
            ';' => {
                result.push(ch);
            }
            _ => {
                if ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
                    label_buffer.push(ch);
                } else if !ch.is_whitespace() {
                    if !label_buffer.is_empty() {
                        result.push_str(&label_buffer);
                        label_buffer.clear();
                    }
                    result.push(ch);
                }
            }
        }
    }
    
    result
}
use crate::clustering::Dendrogram;
use plotters::prelude::*;
use plotters::coord::types::RangedCoordf64;
use std::collections::HashMap;
use std::error::Error;

/// Represents a node in the dendrogram tree structure
#[derive(Debug, Clone)]
struct DendrogramNode {
    left: Option<Box<DendrogramNode>>,
    right: Option<Box<DendrogramNode>>,
    height: f64,
    leaf_index: Option<usize>,
    leaf_count: usize,
}

impl DendrogramNode {
    fn new_leaf(index: usize) -> Self {
        DendrogramNode {
            left: None,
            right: None,
            height: 0.0,
            leaf_index: Some(index),
            leaf_count: 1,
        }
    }
    
    fn new_internal(left: DendrogramNode, right: DendrogramNode, height: f64) -> Self {
        let leaf_count = left.leaf_count + right.leaf_count;
        DendrogramNode {
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            height,
            leaf_index: None,
            leaf_count,
        }
    }
    
    /// Get the center position of this node (for plotting)
    fn get_center(&self, positions: &HashMap<usize, f64>) -> f64 {
        match self.leaf_index {
            Some(idx) => *positions.get(&idx).unwrap_or(&0.0),
            None => {
                let left_center = self.left.as_ref().unwrap().get_center(positions);
                let right_center = self.right.as_ref().unwrap().get_center(positions);
                (left_center + right_center) / 2.0
            }
        }
    }
}

/// Build a tree structure from linkage matrix
fn build_tree(dendrogram: &Dendrogram, n_samples: usize) -> DendrogramNode {
    let mut nodes: HashMap<usize, DendrogramNode> = HashMap::new();
    
    // Initialize leaf nodes
    for i in 0..n_samples {
        nodes.insert(i, DendrogramNode::new_leaf(i));
    }
    
    // Build internal nodes from linkage steps
    for (step_idx, &(cluster1, cluster2, dissimilarity, _size)) in dendrogram.linkage.iter().enumerate() {
        let new_index = n_samples + step_idx;
        
        let left = nodes.remove(&cluster1).unwrap();
        let right = nodes.remove(&cluster2).unwrap();
        
        let internal_node = DendrogramNode::new_internal(left, right, dissimilarity);
        nodes.insert(new_index, internal_node);
    }
    
    // Return the root node
    let root_index = n_samples + dendrogram.linkage.len() - 1;
    nodes.remove(&root_index).unwrap()
}

/// Plot dendrogram to a PNG file
///
/// # Arguments
/// * `dendrogram` - The hierarchical clustering result
/// * `labels` - Sample names corresponding to leaf nodes
/// * `output_path` - Path to save the PNG file
/// * `title` - Title for the plot
pub fn plot_dendrogram(
    dendrogram: &Dendrogram,
    labels: &[String],
    output_path: &str,
    title: &str,
) -> Result<(), Box<dyn Error>> {
    let n_samples = labels.len();
    
    // Build tree structure
    let root = build_tree(dendrogram, n_samples);
    
    // Calculate leaf positions (evenly spaced)
    let mut leaf_positions = HashMap::new();
    for i in 0..n_samples {
        leaf_positions.insert(i, i as f64);
    }
    
    // Find max height for y-axis scaling
    let max_height = dendrogram
        .linkage
        .iter()
        .map(|(_, _, h, _)| *h)
        .fold(0.0, f64::max);
    
    // Create drawing area
    let root_area = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root_area.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root_area)
        .caption(title, ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(150)
        .y_label_area_size(60)
        .build_cartesian_2d(
            -0.5..(n_samples as f64 - 0.5),
            0.0..(max_height * 1.1),
        )?;
    
    chart
        .configure_mesh()
        .y_desc("Distance")
        .x_labels(n_samples)
        .x_label_formatter(&|x| {
            let idx = *x as usize;
            if idx < labels.len() {
                labels[idx].clone()
            } else {
                String::new()
            }
        })
        .draw()?;
    
    // Draw dendrogram lines
    draw_node(&mut chart, &root, &leaf_positions, max_height)?;
    
    root_area.present()?;
    
    Ok(())
}

/// Recursively draw dendrogram nodes
fn draw_node<'a, DB: DrawingBackend>(
    chart: &mut ChartContext<'a, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    node: &DendrogramNode,
    positions: &HashMap<usize, f64>,
    _max_height: f64,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    if node.left.is_none() || node.right.is_none() {
        // Leaf node, nothing to draw
        return Ok(());
    }
    
    let left = node.left.as_ref().unwrap();
    let right = node.right.as_ref().unwrap();
    
    let left_center = left.get_center(positions);
    let right_center = right.get_center(positions);
    
    let left_height = left.height;
    let right_height = right.height;
    let parent_height = node.height;
    
    // Draw vertical line from left child to parent height
    chart.draw_series(LineSeries::new(
        vec![(left_center, left_height), (left_center, parent_height)],
        &BLUE,
    ))?;
    
    // Draw vertical line from right child to parent height
    chart.draw_series(LineSeries::new(
        vec![(right_center, right_height), (right_center, parent_height)],
        &BLUE,
    ))?;
    
    // Draw horizontal line connecting left and right at parent height
    chart.draw_series(LineSeries::new(
        vec![(left_center, parent_height), (right_center, parent_height)],
        &BLUE,
    ))?;
    
    // Recursively draw children
    draw_node(chart, left, positions, _max_height)?;
    draw_node(chart, right, positions, _max_height)?;
    
    Ok(())
}

/// Create a dendrogram plot with default settings
pub fn create_dendrogram(
    dendrogram: &Dendrogram,
    labels: &[String],
    kmer_size: usize,
) -> Result<String, Box<dyn Error>> {
    let output_path = format!("{}_analyzed.png", kmer_size);
    let title = format!("Hierarchical Clustering Dendrogram (k={})", kmer_size);
    
    plot_dendrogram(dendrogram, labels, &output_path, &title)?;
    
    Ok(output_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dendrogram_node_creation() {
        let leaf = DendrogramNode::new_leaf(0);
        assert_eq!(leaf.leaf_index, Some(0));
        assert_eq!(leaf.leaf_count, 1);
        assert_eq!(leaf.height, 0.0);
    }
    
    #[test]
    fn test_internal_node() {
        let left = DendrogramNode::new_leaf(0);
        let right = DendrogramNode::new_leaf(1);
        let internal = DendrogramNode::new_internal(left, right, 1.5);
        
        assert_eq!(internal.leaf_count, 2);
        assert_eq!(internal.height, 1.5);
        assert!(internal.left.is_some());
        assert!(internal.right.is_some());
    }
}
use super::kmer::{Bitpack, Kmer, RevComp, Unpack};
use crate::clustering::{hierarchical_clustering, prepare_data_matrix};
use crate::plotting::create_dendrogram;
use bio::io::fasta;
use dashmap::DashMap;
use fxhash::FxHasher;
use rayon::prelude::*;
use serde_json;
use std::{
    collections::HashMap,
    error::Error,
    fs,
    hash::BuildHasherDefault,
    path::Path,
};

/// A custom `DashMap` w/ `FxHasher`.
///
/// ```use dashmap::DashMap;```
/// ```use fxhash::FxHasher;```
/// ```// skip```
/// ```let dashfx_hash: DashFx = DashMap::with_hasher(BuildHasherDefault::<FxHasher>::default());```
///
/// # Notes
/// Useful: [Using a Custom Hash Function in Rust](https://docs.rs/hashers/1.0.1/hashers/#using-a-custom-hash-function-in-rust)
type DashFx = DashMap<u64, i32, BuildHasherDefault<FxHasher>>;

/// Run complete analysis: k-mer counting, clustering, and plotting on multi-record FASTA file
pub fn run<P: AsRef<Path> + std::fmt::Debug>(fasta_path: P, k: usize) -> Result<(), Box<dyn Error>> {
    eprintln!("Starting k-mer analysis for k={}", k);
    eprintln!("Input FASTA file: {}", fasta_path.as_ref().display());
    
    // Read all records from the FASTA file
    let reader = fasta::Reader::from_file(&fasta_path)?;
    let mut all_kmer_counts: HashMap<String, HashMap<String, i32>> = HashMap::new();
    
    let mut record_count = 0;
    for (idx, result) in reader.records().enumerate() {
        let record = result?;
        let record_id = record.id().to_string();
        
        eprintln!("Processing record {}: {}", idx + 1, record_id);
        
        // Count k-mers for this specific record
        let kmer_counts = count_kmers_from_record(&record, k)?;
        all_kmer_counts.insert(record_id, kmer_counts);
        
        record_count += 1;
    }
    
    if record_count == 0 {
        return Err("No records found in the FASTA file".into());
    }
    
    eprintln!("Found {} records in total", record_count);
    eprintln!("K-mer counting completed. Starting clustering...");
    
    // Prepare data matrix for clustering
    let (data_matrix, sample_names) = prepare_data_matrix(&all_kmer_counts)?;
    eprintln!("Data matrix shape: {} samples x {} features", data_matrix.nrows(), data_matrix.ncols());
    
    // Perform hierarchical clustering
    let dendrogram = hierarchical_clustering(data_matrix)?;
    eprintln!("Clustering completed. Generating dendrogram plot...");
    
    // Create dendrogram plot
    let output_file = create_dendrogram(&dendrogram, &sample_names, k)?;
    eprintln!("Dendrogram saved to: {}", output_file);
    
    // Save results as JSON
    let json_output = format!("K{}.combine.json", k);
    let json_file = fs::File::create(&json_output)?;
    serde_json::to_writer_pretty(json_file, &all_kmer_counts)?;
    eprintln!("K-mer counts saved to: {}", json_output);
    
    Ok(())
}

/// Count k-mers from a single FASTA record
fn count_kmers_from_record(record: &fasta::Record, k: usize) -> Result<HashMap<String, i32>, Box<dyn Error>> {
    let map: DashFx = DashMap::with_hasher(BuildHasherDefault::<FxHasher>::default());
    
    let seq: &[u8] = record.seq();
    process_seq(seq, &k, &map);
    
    let counts: HashMap<String, i32> = map
        .into_iter()
        .par_bridge()
        .map(|(kmer, freq)| (Unpack::bit(kmer, k).0, freq))
        .map(|(kmer, freq)| {
            let kmer = String::from_utf8(kmer).unwrap();
            (kmer, freq)
        })
        .collect();
    
    Ok(counts)
}

/// Ignore substrings containing `N`
///
/// # Notes
/// Canonicalizes by lexicographically smaller of k-mer/reverse-complement
fn process_seq(seq: &[u8], k: &usize, kmer_map: &DashFx) {
    let mut i = 0;

    while i <= seq.len() - k {
        let sub = &seq[i..i + k];

        if let Ok(kmer) = Kmer::from_sub(sub) {
            process_valid_bytes(kmer_map, kmer);

            i += 1;
        } else {
            let invalid_byte_index = Kmer::find_invalid(sub);

            i += invalid_byte_index + 1;
        }
    }
}

/// Convert a valid sequence substring from a bytes string to a u64
fn process_valid_bytes(kmer_map: &DashFx, kmer: Kmer) {
    let Bitpack(bitpacked_kmer) = kmer.0.iter().cloned().collect();

    // If the k-mer as found in the sequence is already a key in the `Dashmap`,
    // increment its value and move on
    if let Some(mut freq) = kmer_map.get_mut(&bitpacked_kmer) {
        *freq += 1;
    } else {
        // Initialize the reverse complement of this so-far unrecorded k-mer
        let RevComp(revcompkmer) = RevComp::from_kmer(&kmer);

        // Find the alphabetically less of the k-mer substring and its reverse complement
        let canonical_kmer = Kmer::canonical(revcompkmer, kmer.0);

        // Compress the canonical k-mer into a bitpacked 64-bit unsigned integer
        let kmer: Bitpack = canonical_kmer.collect();

        // Add k-mer key and initial value to results
        *kmer_map.entry(kmer.0).or_insert(0) += 1;
    }
}
