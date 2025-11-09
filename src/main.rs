use bio::io::fasta;
use clap::Parser;
use dbscan::Classification;
use itertools::Itertools;
use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::process;

/// CLI tool for clustering contigs based on k-mer spatial distribution
#[derive(Parser, Debug)]
#[command(name = "kalnal")]
#[command(about = "Cluster contigs using k-mer interval histograms and DBSCAN", long_about = None)]
struct Args {
    /// K-mer length
    k: usize,

    /// Input FASTA file path
    fasta_file: String,

    /// Output TSV file path
    output_file: String,

    /// Number of k-mers to sample
    #[arg(short = 'n', long = "n-kmers", default_value_t = 1000)]
    n_kmers: usize,

    /// DBSCAN epsilon parameter (maximum distance for neighborhood)
    /// If not specified, will be auto-detected as 2*D where D is dimensionality
    #[arg(short = 'e', long = "eps")]
    eps: Option<f64>,

    /// DBSCAN minimum points parameter (minimum neighbors to form a cluster)
    /// If not specified, will be auto-detected using k-NN elbow method
    #[arg(short = 'm', long = "min-points")]
    min_points: Option<usize>,
}

// Log-scale histogram bins (powers of 4, up to ~1G)
const BINS: &[usize] = &[
    0, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 
    16777216, 67108864, 268435456, usize::MAX
];

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
    // PASS 1: scan FASTA to count k-mer frequencies
    eprintln!("Counting k-mer frequencies (k={})...", args.k);
    if args.k == 0 || args.k > 32 {
        return Err("k must be between 1 and 32 when packing into u64".into());
    }
    let kmer_counts = count_kmers(&contigs, args.k);
    eprintln!("Found {} unique canonical k-mers", kmer_counts.len());
    if kmer_counts.is_empty() {
        return Err("No valid k-mers found".into());
    }

    // Select most frequent n_kmers
    let n_sample = args.n_kmers.min(kmer_counts.len());
    eprintln!("Selecting {} most frequent k-mers...", n_sample);
    let selected_kmers = select_top_frequent_kmers(&kmer_counts, n_sample);
    eprintln!("Selected k-mers with frequencies ranging from {} to {}", 
        kmer_counts.get(&selected_kmers[selected_kmers.len()-1]).unwrap_or(&0),
        kmer_counts.get(&selected_kmers[0]).unwrap_or(&0));

    eprintln!("Scanning FASTA and building position index for sampled k-mers...");

    // PASS 2: scan FASTA again and only record positions for sampled k-mers
    let sampled_set: HashSet<u64> = selected_kmers.iter().copied().collect();
    let kmer_index = build_kmer_position_index_sampled(&contigs, args.k, &sampled_set);
    eprintln!("Indexed positions for {} sampled k-mers", kmer_index.len());

    eprintln!("Building cosine distance matrix from high-dimensional feature space...");
    
    // Build cosine distance matrix from concatenated normalized histograms
    let averaged_dist_matrix = build_cosine_distance_matrix(&selected_kmers, &kmer_index, contigs.len(), BINS);
    
    // Auto-detect DBSCAN parameters if not provided
    let (eps, min_points) = if args.eps.is_none() || args.min_points.is_none() {
        eprintln!("Auto-detecting DBSCAN parameters using k-NN method...");
        let (auto_eps, auto_min_points) = auto_detect_dbscan_params(&averaged_dist_matrix);
        let final_eps = args.eps.unwrap_or(auto_eps);
        let final_min_points = args.min_points.unwrap_or(auto_min_points);
        eprintln!("Auto-detected: eps={:.2}, min_points={}", auto_eps, auto_min_points);
        if args.eps.is_some() {
            eprintln!("Using user-provided eps={:.2}", final_eps);
        }
        if args.min_points.is_some() {
            eprintln!("Using user-provided min_points={}", final_min_points);
        }
        (final_eps, final_min_points)
    } else {
        (args.eps.unwrap(), args.min_points.unwrap())
    };
    
    eprintln!("Running DBSCAN clustering (eps={:.2}, min_points={})...", eps, min_points);
    
    // Perform DBSCAN clustering
    let clusters = perform_dbscan(&averaged_dist_matrix, eps, min_points);
    
    eprintln!("Clustering complete. Writing results to {}...", args.output_file);
    
    // Write results to TSV
    let mut output = File::create(&args.output_file)?;
    writeln!(output, "contig_id\tcluster_id")?;
    
    for (i, contig_id) in contig_ids.iter().enumerate() {
        let cluster_label = match &clusters[i] {
            Classification::Core(cluster_id) | Classification::Edge(cluster_id) => {
                cluster_id.to_string()
            }
            Classification::Noise => "noise".to_string(),
        };
        writeln!(output, "{}\t{}", contig_id, cluster_label)?;
    }

    eprintln!("Success! Clustering results written to {}", args.output_file);
    
    // Print summary
    let mut cluster_counts: HashMap<String, usize> = HashMap::new();
    for cluster in &clusters {
        let label = match cluster {
            Classification::Core(id) | Classification::Edge(id) => id.to_string(),
            Classification::Noise => "noise".to_string(),
        };
        *cluster_counts.entry(label).or_insert(0) += 1;
    }
    
    eprintln!("\nCluster summary:");
    let mut sorted_clusters: Vec<_> = cluster_counts.iter().collect();
    sorted_clusters.sort_by_key(|(label, _)| {
        if label.as_str() == "noise" {
            (usize::MAX, label.to_string())
        } else {
            (label.parse::<usize>().unwrap_or(usize::MAX), label.to_string())
        }
    });
    
    for (cluster_id, count) in sorted_clusters {
        eprintln!("  Cluster {}: {} contigs", cluster_id, count);
    }
    
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

/// PASS 1: count k-mer frequencies (packed u64)
fn count_kmers(contigs: &[(String, Vec<u8>)], k: usize) -> HashMap<u64, usize> {
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for (_id, seq) in contigs.iter() {
        for window in seq.windows(k) {
            if let Some(key) = canonical_kmer_u64(window) {
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    counts
}

/// Select top n k-mers from Q2-Q3 (50%-75%) frequency range
fn select_top_frequent_kmers(kmer_counts: &HashMap<u64, usize>, n: usize) -> Vec<u64> {
    // Collect all frequency values
    let mut counts: Vec<usize> = kmer_counts.values().copied().collect();
    
    if counts.is_empty() {
        return Vec::new();
    }
    
    // Sort frequencies
    counts.sort_unstable();

    // Calculate Q1 (25th percentile, median) and Q3 (75th percentile)
    let q1_idx = ((counts.len() as f64) * 0.25) as usize;
    let q3_idx = ((counts.len() as f64) * 0.75) as usize;

    let q1_freq = counts.get(q1_idx).copied().unwrap_or(0);
    let q3_freq = counts.get(q3_idx).copied().unwrap_or(usize::MAX);
    
    // Filter k-mers in Q2-Q3 range
    let mut filtered_kmers: Vec<(u64, usize)> = kmer_counts
        .iter()
        .filter(|(_, count)| **count >= q1_freq && **count <= q3_freq)
        .map(|(&kmer, &count)| (kmer, count))
        .collect();
    
    // Sort by frequency descending
    filtered_kmers.sort_by(|a, b| b.1.cmp(&a.1));
    
    // Take top n without random sampling
    filtered_kmers.into_iter().take(n).map(|(kmer, _)| kmer).collect()
}

/// PASS 2: build k-mer position index only for sampled k-mers
/// Returns: HashMap<packed_kmer_u64, Vec<(contig_idx as u32, pos as u32)>>
fn build_kmer_position_index_sampled(
    contigs: &[(String, Vec<u8>)],
    k: usize,
    sampled: &HashSet<u64>,
) -> HashMap<u64, Vec<(u32, u32)>> {
    let mut index: HashMap<u64, Vec<(u32, u32)>> = HashMap::new();

    for (contig_idx, (_id, seq)) in contigs.iter().enumerate() {
        for (pos, window) in seq.windows(k).enumerate() {
            if let Some(key) = canonical_kmer_u64(window) {
                if !sampled.contains(&key) {
                    continue;
                }

                let entry = index.entry(key).or_insert_with(Vec::new);
                entry.push((contig_idx as u32, pos as u32));
            }
        }
    }

    index
}

/// Calculate interval histogram using precomputed k-mer position index (packed u64 keys)
/// Returns L1-normalized histogram per contig
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

    // L1 normalization per contig
    for contig_idx in 0..n_contigs {
        let start = contig_idx * n_bins;
        let end = (contig_idx + 1) * n_bins;
        
        // Calculate sum for this contig's histogram
        let sum: f64 = histogram.slice(ndarray::s![start..end]).sum();
        
        // Normalize if sum > 0
        if sum > 0.0 {
            let mut slice = histogram.slice_mut(ndarray::s![start..end]);
            slice /= sum;
        }
    }

    histogram
}

/// Build cosine distance matrix from high-dimensional feature matrix
/// Concatenates normalized histograms from all selected k-mers into a single feature space
fn build_cosine_distance_matrix(
    selected_kmers: &[u64],
    kmer_index: &HashMap<u64, Vec<(u32, u32)>>,
    n_contigs: usize,
    bins: &[usize],
) -> Array2<f64> {
    let n_bins = bins.len() - 1;
    let n_features = selected_kmers.len() * n_bins;
    
    // Create high-dimensional feature matrix (n_contigs x n_features)
    let mut feature_matrix = Array2::<f64>::zeros((n_contigs, n_features));
    
    // Fill feature matrix with normalized histograms from each k-mer
    for (k_idx, &kmer_u64) in selected_kmers.iter().enumerate() {
        // Get normalized histogram for this k-mer
        let norm_hist = calculate_interval_histogram_from_index_u64(
            kmer_u64,
            kmer_index,
            n_contigs,
            bins,
        );
        
        // Copy each contig's histogram slice into feature matrix
        for contig_idx in 0..n_contigs {
            let hist_start = contig_idx * n_bins;
            let hist_end = (contig_idx + 1) * n_bins;
            let feat_start = k_idx * n_bins;
            let feat_end = (k_idx + 1) * n_bins;
            
            let hist_slice = norm_hist.slice(ndarray::s![hist_start..hist_end]);
            let mut feat_slice = feature_matrix.slice_mut(ndarray::s![contig_idx, feat_start..feat_end]);
            feat_slice.assign(&hist_slice);
        }
    }
    
    // Build cosine distance matrix
    let mut dist_matrix = Array2::<f64>::zeros((n_contigs, n_contigs));
    
    for i in 0..n_contigs {
        for j in (i + 1)..n_contigs {
            let vec_i = feature_matrix.row(i);
            let vec_j = feature_matrix.row(j);
            
            // Calculate cosine similarity
            let dot_product = vec_i.dot(&vec_j);
            let norm_i = vec_i.dot(&vec_i).sqrt();
            let norm_j = vec_j.dot(&vec_j).sqrt();
            let denominator = norm_i * norm_j;
            
            let sim = if denominator.abs() < f64::EPSILON {
                0.0
            } else {
                dot_product / denominator
            };
            
            // Cosine distance = 1 - cosine similarity
            let dist = 1.0 - sim;
            
            dist_matrix[[i, j]] = dist;
            dist_matrix[[j, i]] = dist;
        }
    }
    
    dist_matrix
}

/// Auto-detect DBSCAN parameters using k-NN distance analysis
/// Returns (eps, min_points) where:
/// - eps is determined from k-distance graph knee detection only
/// - min_points is optimized using elbow method
fn auto_detect_dbscan_params(dist_matrix: &Array2<f64>) -> (f64, usize) {
    let n = dist_matrix.nrows();
    
    // Calculate k-NN distances for each point
    let mut knn_distances: Vec<Vec<f64>> = Vec::new();
    
    for i in 0..n {
        let mut distances: Vec<f64> = Vec::new();
        for j in 0..n {
            if i != j {
                distances.push(dist_matrix[[i, j]]);
            }
        }
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        knn_distances.push(distances);
    }
    
    // Step 1: Determine optimal k for min_points using elbow method
    let max_k = n - 1;
    
    // Calculate average k-NN distance for each k
    let mut avg_knn_dist: Vec<f64> = Vec::new();
    for k in 1..=max_k {
        let sum: f64 = knn_distances.iter()
            .map(|dists| if k-1 < dists.len() { dists[k-1] } else { 0.0 })
            .sum();
        avg_knn_dist.push(sum / n as f64);
    }
    
    // Find elbow point using rate of change
    let elbow_k = find_elbow_point(&avg_knn_dist);
    let min_points = elbow_k;
    
    eprintln!("  k-NN elbow point detected at k={}", elbow_k);
    eprintln!("  Using elbow-based min_points: {}", min_points);
    
    // Step 2: Determine eps using k-distance graph knee detection
    // For eps, we use the k-distance where k = min_points
    let k_for_eps = min_points.min(max_k);
    
    // Get k-distances (k-th nearest neighbor distance) for all points
    let mut k_distances: Vec<f64> = knn_distances.iter()
        .filter_map(|dists| {
            if k_for_eps - 1 < dists.len() {
                Some(dists[k_for_eps - 1])
            } else {
                None
            }
        })
        .collect();
    
    k_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    if k_distances.is_empty() {
        eprintln!("  Warning: No valid k-distances found, using fallback eps=0.1");
        return (0.1, min_points);
    }
    
    // Find knee point in sorted k-distances using maximum distance method
    let eps = find_knee_in_sorted_distances(&k_distances);
    
    eprintln!("  k-distance knee point (k={}): {:.2}", k_for_eps, eps);
    eprintln!("  Selected eps={:.2}, min_points={}", eps, min_points);
    
    (eps, min_points)
}

/// Find knee point in sorted k-distances using maximum distance from line method
fn find_knee_in_sorted_distances(sorted_distances: &[f64]) -> f64 {
    if sorted_distances.len() < 3 {
        return sorted_distances.last().copied().unwrap_or(1.0);
    }
    
    let n = sorted_distances.len();
    
    // Normalize coordinates
    let x_start = 0.0;
    let y_start = sorted_distances[0];
    let x_end = (n - 1) as f64;
    let y_end = sorted_distances[n - 1];
    
    // If all distances are the same, return that value
    if (y_end - y_start).abs() < 1e-10 {
        return y_start;
    }
    
    // Calculate perpendicular distance from each point to the line
    // Line equation: (y_end - y_start) * x - (x_end - x_start) * y + (x_end * y_start - x_start * y_end) = 0
    let a = y_end - y_start;
    let b = -(x_end - x_start);
    let c = x_end * y_start - x_start * y_end;
    let norm = (a * a + b * b).sqrt();
    
    let mut max_distance = 0.0;
    let mut knee_idx = n - 1;
    
    for i in 0..n {
        let x = i as f64;
        let y = sorted_distances[i];
        
        // Perpendicular distance from point (x, y) to line
        let distance = (a * x + b * y + c).abs() / norm;
        
        if distance > max_distance {
            max_distance = distance;
            knee_idx = i;
        }
    }
    
    sorted_distances[knee_idx]
}

/// Find elbow point in a curve using maximum curvature
fn find_elbow_point(values: &[f64]) -> usize {
    if values.len() < 3 {
        return values.len();
    }
    
    // Normalize values to [0, 1] range
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    
    if range < 1e-10 {
        return values.len();
    }
    
    let normalized: Vec<f64> = values.iter()
        .map(|v| (v - min_val) / range)
        .collect();
    
    // Calculate curvature at each point using discrete approximation
    // Curvature = |y''| / (1 + y'^2)^(3/2)
    let mut max_curvature = 0.0;
    let mut elbow_idx = 1;
    
    for i in 1..(normalized.len() - 1) {
        // First derivative (finite difference)
        let dy1 = normalized[i] - normalized[i - 1];
        let dy2 = normalized[i + 1] - normalized[i];
        
        // Second derivative
        let d2y = dy2 - dy1;
        
        // Approximate curvature
        let curvature = d2y.abs();
        
        if curvature > max_curvature {
            max_curvature = curvature;
            elbow_idx = i;
        }
    }
    
    // Return k value (1-indexed)
    elbow_idx + 1
}

/// Perform DBSCAN clustering on distance matrix
fn perform_dbscan(dist_matrix: &Array2<f64>, eps: f64, min_points: usize) -> Vec<Classification> {
    let n = dist_matrix.nrows();
    
    // Convert distance matrix to feature vectors (each row as a point)
    let mut points: Vec<Vec<f64>> = Vec::new();
    for i in 0..n {
        let mut row = Vec::new();
        for j in 0..n {
            row.push(dist_matrix[[i, j]]);
        }
        points.push(row);
    }
    
    let model = dbscan::Model::new(eps, min_points);
    model.run(&points)
}
