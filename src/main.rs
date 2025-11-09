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
    
    // STEP 1: Build k-mer position index (scan once)
    eprintln!("Building k-mer position index (k={})...", args.k);
    let kmer_index = build_kmer_position_index(&contigs, args.k);
    eprintln!("Indexed {} unique k-mers", kmer_index.len());

    if kmer_index.is_empty() {
        return Err("No valid k-mers found".into());
    }

    // STEP 2: Sample k-mers
    let n_sample = args.n_kmers.min(kmer_index.len());
    eprintln!("Randomly sampling {} k-mers...", n_sample);

    let mut rng = thread_rng();
    let all_kmers: Vec<Vec<u8>> = kmer_index.keys().cloned().collect();
    let selected_kmers: Vec<Vec<u8>> = all_kmers
        .choose_multiple(&mut rng, n_sample)
        .cloned()
        .collect();

    eprintln!("Building tree for each k-mer and collecting bipartitions...");
    
    // STEP 3: Build trees using precomputed index (fast lookups)
    let all_bipartitions: Vec<BTreeSet<BTreeSet<String>>> = selected_kmers
        .par_iter()
        .enumerate()
        .map(|(i, kmer)| {
            if i % 100 == 0 && i > 0 {
                eprintln!("  Processed {}/{} k-mers", i, n_sample);
            }
            
            // Fast lookup from index (no file scanning!)
            let histogram = calculate_interval_histogram_from_index(
                kmer, 
                &kmer_index, 
                contigs.len(), 
                BINS
            );
            
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

/// Build k-mer position index: scan genome once and record all k-mer positions
/// Returns: HashMap<Kmer, Vec<(ContigIndex, Position)>>
fn build_kmer_position_index(
    contigs: &[(String, Vec<u8>)],
    k: usize,
) -> HashMap<Vec<u8>, Vec<(usize, usize)>> {
    let mut index: HashMap<Vec<u8>, Vec<(usize, usize)>> = HashMap::new();
    
    for (contig_idx, (_id, seq)) in contigs.iter().enumerate() {
        for (pos, window) in seq.windows(k).enumerate() {
            // Skip k-mers containing N
            if window.iter().any(|&b| b == b'N' || b == b'n') {
                continue;
            }
            
            let canonical = canonical_kmer(window);
            index.entry(canonical).or_insert_with(Vec::new).push((contig_idx, pos));
        }
    }
    
    index
}

/// Calculate interval histogram using precomputed k-mer position index (fast lookup!)
fn calculate_interval_histogram_from_index(
    kmer: &[u8],
    kmer_index: &HashMap<Vec<u8>, Vec<(usize, usize)>>,
    n_contigs: usize,
    bins: &[usize],
) -> Array1<f64> {
    let n_bins = bins.len() - 1;
    let mut histogram = Array1::<f64>::zeros(n_contigs * n_bins);
    
    // Fast lookup from index
    if let Some(positions) = kmer_index.get(kmer) {
        // Group positions by contig
        let mut contig_positions: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(contig_idx, pos) in positions {
            contig_positions.entry(contig_idx).or_insert_with(Vec::new).push(pos);
        }
        
        // Calculate intervals for each contig
        for (contig_idx, mut pos_list) in contig_positions {
            pos_list.sort_unstable();
            
            // Calculate intervals between consecutive positions
            let intervals: Vec<usize> = pos_list
                .iter()
                .tuple_windows()
                .map(|(a, b)| b - a)
                .collect();
            
            // Bin the intervals
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
