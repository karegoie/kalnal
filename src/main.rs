use bio::io::fasta;
use clap::Parser;
use itertools::Itertools;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet};
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
    /// Maximum positions to store per k-mer (safety cutoff to avoid memory explosion)
    #[arg(long = "max-kmer-freq", default_value_t = 100000)]
    max_kmer_freq: usize,
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
    
    // TWO-PASS STRATEGY
    // PASS 1: scan FASTA to collect unique canonical k-mers (no positions)
    eprintln!("PASS 1: collecting unique canonical k-mers (k={})...", args.k);
    if args.k == 0 || args.k > 32 {
        return Err("k must be between 1 and 32 when packing into u64".into());
    }
    let unique_kmers = collect_unique_kmers(&contigs, args.k);
    eprintln!("Found {} unique canonical k-mers", unique_kmers.len());
    if unique_kmers.is_empty() {
        return Err("No valid k-mers found".into());
    }

    // Sampling: choose n_kmers from unique set
    let n_sample = args.n_kmers.min(unique_kmers.len());
    eprintln!("Sampling {} k-mers from the unique set...", n_sample);
    let mut rng = thread_rng();
    let mut all_kmers_vec: Vec<u64> = unique_kmers.into_iter().collect();
    let selected_kmers: Vec<u64> = all_kmers_vec
        .as_mut_slice()
        .choose_multiple(&mut rng, n_sample)
        .cloned()
        .collect();

    eprintln!("PASS 2: scanning FASTA and building position index for sampled k-mers (max per k-mer={})...", args.max_kmer_freq);

    // PASS 2: scan FASTA again and only record positions for sampled k-mers
    let sampled_set: HashSet<u64> = selected_kmers.iter().copied().collect();
    let kmer_index = build_kmer_position_index_sampled(&contigs, args.k, &sampled_set, args.max_kmer_freq);
    eprintln!("Indexed positions for {} sampled k-mers", kmer_index.len());

    eprintln!("Building tree for each sampled k-mer and collecting bipartitions...");
    // STEP 3: Build trees using precomputed index (fast lookups)
    let all_bipartitions: Vec<BTreeSet<BTreeSet<String>>> = selected_kmers
        .par_iter()
        .enumerate()
        .map(|(i, kmer_u64)| {
            if i % 100 == 0 && i > 0 {
                eprintln!("  Processed {}/{} k-mers", i, n_sample);
            }

            let histogram = calculate_interval_histogram_from_index_u64(
                *kmer_u64,
                &kmer_index,
                contigs.len(),
                BINS,
            );

            let dist_matrix = histogram_to_distance_matrix(&histogram, BINS.len() - 1);
            let tree_newick = build_nj_tree_simple(&dist_matrix, &contig_ids);
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

    // Build consensus tree: use the first sampled k-mer's histogram (from index)
    let first_histogram = calculate_interval_histogram_from_index_u64(selected_kmers[0], &kmer_index, contigs.len(), BINS);
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

/// Pack a k-mer window into a canonical u64 (2 bits per base). Returns None if contains N or k>32.
fn canonical_kmer_u64(window: &[u8]) -> Option<u64> {
    // Map A/ a -> 0, C -> 1, G -> 2, T -> 3
    let k = window.len();
    if k == 0 || k > 32 {
        return None;
    }

    let mut fwd: u64 = 0;
    let mut rev: u64 = 0;
    for &b in window.iter() {
        let v = match b {
            b'A' | b'a' => 0u64,
            b'C' | b'c' => 1u64,
            b'G' | b'g' => 2u64,
            b'T' | b't' => 3u64,
            _ => return None,
        };
        fwd = (fwd << 2) | v;
        // For reverse complement, insert complement at low bits
        let cv = match v {
            0 => 3u64, // A -> T
            1 => 2u64, // C -> G
            2 => 1u64, // G -> C
            3 => 0u64, // T -> A
            _ => unreachable!(),
        };
        rev = rev | (cv << (2 * (window.len() - 1)));
        // shift existing rev right by 2 for next iteration
        // but we build differently: we'll construct rev by shifting existing bits right
        // Simpler: compute rev by iterating reversed below instead
    }

    // build reverse complement properly
    let mut rc: u64 = 0;
    for &b in window.iter().rev() {
        let v = match b {
            b'A' | b'a' => 0u64,
            b'C' | b'c' => 1u64,
            b'G' | b'g' => 2u64,
            b'T' | b't' => 3u64,
            _ => return None,
        };
        let cv = match v {
            0 => 3u64,
            1 => 2u64,
            2 => 1u64,
            3 => 0u64,
            _ => unreachable!(),
        };
        rc = (rc << 2) | cv;
    }

    Some(std::cmp::min(fwd, rc))
}

/// PASS 1: collect unique canonical k-mers (packed u64)
fn collect_unique_kmers(contigs: &[(String, Vec<u8>)], k: usize) -> HashSet<u64> {
    let mut set: HashSet<u64> = HashSet::new();
    for (_id, seq) in contigs.iter() {
        for window in seq.windows(k) {
            if let Some(key) = canonical_kmer_u64(window) {
                set.insert(key);
            }
        }
    }
    set
}

/// PASS 2: build k-mer position index only for sampled k-mers
/// Returns: HashMap<packed_kmer_u64, Vec<(contig_idx as u32, pos as u32)>>
fn build_kmer_position_index_sampled(
    contigs: &[(String, Vec<u8>)],
    k: usize,
    sampled: &HashSet<u64>,
    max_freq: usize,
) -> HashMap<u64, Vec<(u32, u32)>> {
    let mut index: HashMap<u64, Vec<(u32, u32)>> = HashMap::new();
    let mut saturated: HashSet<u64> = HashSet::new();

    for (contig_idx, (_id, seq)) in contigs.iter().enumerate() {
        for (pos, window) in seq.windows(k).enumerate() {
            if let Some(key) = canonical_kmer_u64(window) {
                if saturated.contains(&key) {
                    continue;
                }
                if !sampled.contains(&key) {
                    continue;
                }

                let entry = index.entry(key).or_insert_with(Vec::new);
                if entry.len() >= max_freq {
                    // mark saturated and drop further additions
                    saturated.insert(key);
                    // optionally free memory or shrink
                    continue;
                }
                entry.push((contig_idx as u32, pos as u32));
                if entry.len() >= max_freq {
                    saturated.insert(key);
                }
            }
        }
    }

    index
}

/// Calculate interval histogram using precomputed k-mer position index (packed u64 keys)
fn calculate_interval_histogram_from_index_u64(
    kmer_u64: u64,
    kmer_index: &HashMap<u64, Vec<(u32, u32)>>,
    n_contigs: usize,
    bins: &[usize],
) -> Array1<f64> {
    let n_bins = bins.len() - 1;
    let mut histogram = Array1::<f64>::zeros(n_contigs * n_bins);

    if let Some(positions) = kmer_index.get(&kmer_u64) {
        let mut contig_positions: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(contig_idx_u32, pos_u32) in positions {
            let contig_idx = contig_idx_u32 as usize;
            let pos = pos_u32 as usize;
            contig_positions.entry(contig_idx).or_insert_with(Vec::new).push(pos);
        }

        for (contig_idx, mut pos_list) in contig_positions {
            pos_list.sort_unstable();
            let intervals: Vec<usize> = pos_list
                .iter()
                .tuple_windows()
                .map(|(a, b)| b - a)
                .collect();

            for interval in intervals {
                for bin_idx in 0..n_bins {
                    if interval >= bins[bin_idx] && interval < bins[bin_idx + 1] {
                        histogram[contig_idx * n_bins + bin_idx] += 1.0;
                        break;
                    }
                }
            }
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
