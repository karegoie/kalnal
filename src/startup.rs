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

/// Run complete analysis: k-mer counting, clustering, and plotting
pub fn run(input_dir: &str, k: usize) -> Result<(), Box<dyn Error>> {
    eprintln!("Starting k-mer analysis for k={}", k);
    eprintln!("Input directory: {}", input_dir);
    
    // Find all .split.fa files in the directory
    let paths = fs::read_dir(input_dir)?;
    let mut split_files = Vec::new();
    
    for path in paths {
        let path = path?.path();
        if let Some(filename) = path.file_name() {
            if let Some(name) = filename.to_str() {
                if name.ends_with(".split.fa") {
                    split_files.push(path);
                }
            }
        }
    }
    
    if split_files.is_empty() {
        return Err(format!("No .split.fa files found in {}", input_dir).into());
    }
    
    eprintln!("Found {} split files", split_files.len());
    
    // Count k-mers for each file
    let mut all_kmer_counts: HashMap<String, HashMap<String, i32>> = HashMap::new();
    
    for (idx, file_path) in split_files.iter().enumerate() {
        let sample_name = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&format!("sample_{}", idx))
            .replace(".split", "");
        
        eprintln!("Processing sample {}/{}: {}", idx + 1, split_files.len(), sample_name);
        
        let kmer_counts = count_kmers(file_path, k)?;
        all_kmer_counts.insert(sample_name, kmer_counts);
    }
    
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

/// Count k-mers in a single fasta file and return as HashMap
fn count_kmers<P: AsRef<Path> + std::fmt::Debug>(
    path: P,
    k: usize,
) -> Result<HashMap<String, i32>, Box<dyn Error>> {
    let kmer_map = build_map(path, k)?;
    
    let counts: HashMap<String, i32> = kmer_map
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

/// Reads sequences from fasta records in parallel using [`rayon`](https://docs.rs/rayon/1.5.1/rayon/),
/// using a customized [`dashmap`](https://docs.rs/dashmap/4.0.2/dashmap/struct.DashMap.html)
/// with [`FxHasher`](https://docs.rs/fxhash/0.2.1/fxhash/struct.FxHasher.html) to update in parallel a
/// hashmap of canonical k-mers (keys) and their frequency in the data (values)
fn build_map<P: AsRef<Path> + std::fmt::Debug>(
    path: P,
    k: usize,
) -> Result<DashFx, Box<dyn Error>> {
    let map: DashFx = DashMap::with_hasher(BuildHasherDefault::<FxHasher>::default());

    fasta::Reader::from_file(path)?
        .records()
        .into_iter()
        .par_bridge()
        .for_each(|r| {
            let record = r.expect("Error reading fasta record.");

            let seq: &[u8] = record.seq();

            process_seq(seq, &k, &map);
        });

    Ok(map)
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
