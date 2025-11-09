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
