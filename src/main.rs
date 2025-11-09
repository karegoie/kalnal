use bio::io::fasta;
use clap::Parser;
use dbscan::Classification;
use itertools::Itertools;
use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::process;

#[derive(Parser, Debug)]
#[command(name = "kalnal")]
#[command(about = "Cluster contigs using k-mer interval histograms and DBSCAN", long_about = None)]
struct Args {
    k: usize,
    fasta_file: String,
    output_file: String,
    #[arg(short = 'n', long = "n-kmers", default_value_t = 1000)]
    n_kmers: usize,
    #[arg(short = 'e', long = "eps")]
    eps: Option<f64>,
    #[arg(short = 'm', long = "min-points")]
    min_points: Option<usize>,
}

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
    let reader = fasta::Reader::from_file(&args.fasta_file)?;
    let mut contigs: Vec<(String, Vec<u8>)> = Vec::new();
    let mut contig_ids: Vec<String> = Vec::new();
    for result in reader.records() {
        let record = result?;
        let id = record.id().to_string();
        contig_ids.push(id.clone());
        contigs.push((id, record.seq().to_vec()));
    }
    if contigs.is_empty() {
        return Err("No contigs found in input file".into());
    }
    eprintln!("Loaded {} contigs", contigs.len());

    eprintln!("Counting k-mer frequencies (k={})...", args.k);
    if args.k == 0 || args.k > 32 {
        return Err("k must be between 1 and 32 when packing into u64".into());
    }
    let kmer_counts = count_kmers(&contigs, args.k);
    eprintln!("Found {} unique canonical k-mers", kmer_counts.len());
    if kmer_counts.is_empty() {
        return Err("No valid k-mers found".into());
    }

    let n_sample = args.n_kmers.min(kmer_counts.len());
    eprintln!("Selecting {} most frequent k-mers...", n_sample);
    let selected_kmers = select_top_frequent_kmers(&kmer_counts, n_sample);
    eprintln!("Selected k-mers with frequencies ranging from {} to {}", 
        kmer_counts.get(&selected_kmers[selected_kmers.len()-1]).unwrap_or(&0),
        kmer_counts.get(&selected_kmers[0]).unwrap_or(&0));

    eprintln!("Scanning FASTA and building position index for sampled k-mers...");
    let sampled_set: HashSet<u64> = selected_kmers.iter().copied().collect();
    let kmer_index = build_kmer_position_index_sampled(&contigs, args.k, &sampled_set);
    eprintln!("Indexed positions for {} sampled k-mers", kmer_index.len());

    eprintln!("Building cosine distance matrix from high-dimensional feature space...");
    let dist_matrix = build_cosine_distance_matrix(&selected_kmers, &kmer_index, contigs.len(), BINS);
    
    let (eps, min_points) = if args.eps.is_none() || args.min_points.is_none() {
        eprintln!("Auto-detecting DBSCAN parameters using k-NN method...");
        let (auto_eps, auto_min_points) = auto_detect_dbscan_params(&dist_matrix);
        let final_eps = args.eps.unwrap_or(auto_eps);
        let final_min_points = args.min_points.unwrap_or(auto_min_points);
        eprintln!("Auto-detected: eps={:.2}, min_points={}", auto_eps, auto_min_points);
        if args.eps.is_some() { eprintln!("Using user-provided eps={:.2}", final_eps); }
        if args.min_points.is_some() { eprintln!("Using user-provided min_points={}", final_min_points); }
        (final_eps, final_min_points)
    } else {
        (args.eps.unwrap(), args.min_points.unwrap())
    };
    
    eprintln!("Running DBSCAN clustering (eps={:.2}, min_points={})...", eps, min_points);
    let model = dbscan::Model::new(eps, min_points);
    let clusters = model.run(
        &(0..dist_matrix.nrows())
            .map(|i| dist_matrix.row(i).to_vec())
            .collect::<Vec<_>>()
    );
    
    eprintln!("Clustering complete. Writing results to {}...", args.output_file);
    let mut output = File::create(&args.output_file)?;
    writeln!(output, "contig_id\tcluster_id")?;
    for (i, contig_id) in contig_ids.iter().enumerate() {
        let cluster_label = match &clusters[i] {
            Classification::Core(id) | Classification::Edge(id) => id.to_string(),
            Classification::Noise => "noise".to_string(),
        };
        writeln!(output, "{}\t{}", contig_id, cluster_label)?;
    }
    eprintln!("Success! Clustering results written to {}", args.output_file);
    
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
        if label.as_str() == "noise" { (usize::MAX, label.to_string()) } 
        else { (label.parse::<usize>().unwrap_or(usize::MAX), label.to_string()) }
    });
    for (cluster_id, count) in sorted_clusters {
        eprintln!("  Cluster {}: {} contigs", cluster_id, count);
    }
    
    Ok(())
}

fn canonical_kmer_u64(window: &[u8]) -> Option<u64> {
    let k = window.len();
    if k == 0 || k > 32 { return None; }
    let mut fwd: u64 = 0;
    let mut rc: u64 = 0;
    for &b in window.iter() {
        let v = match b {
            b'A' | b'a' => 0, b'C' | b'c' => 1,
            b'G' | b'g' => 2, b'T' | b't' => 3,
            _ => return None,
        };
        fwd = (fwd << 2) | v;
    }
    for &b in window.iter().rev() {
        let v = match b {
            b'A' | b'a' => 0, b'C' | b'c' => 1,
            b'G' | b'g' => 2, b'T' | b't' => 3,
            _ => return None,
        };
        rc = (rc << 2) | (3 - v);
    }
    Some(std::cmp::min(fwd, rc))
}

fn count_kmers(contigs: &[(String, Vec<u8>)], k: usize) -> HashMap<u64, usize> {
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for (_, seq) in contigs.iter() {
        for window in seq.windows(k) {
            if let Some(key) = canonical_kmer_u64(window) {
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    counts
}

fn select_top_frequent_kmers(kmer_counts: &HashMap<u64, usize>, n: usize) -> Vec<u64> {
    let mut counts: Vec<usize> = kmer_counts.values().copied().collect();
    if counts.is_empty() { return Vec::new(); }
    counts.sort_unstable();
    let q1_idx = (counts.len() as f64 * 0.25) as usize;
    let q3_idx = (counts.len() as f64 * 0.75) as usize;
    let q1_freq = counts.get(q1_idx).copied().unwrap_or(0);
    let q3_freq = counts.get(q3_idx).copied().unwrap_or(usize::MAX);
    let mut filtered_kmers: Vec<(u64, usize)> = kmer_counts.iter()
        .filter(|(_, count)| **count >= q1_freq && **count <= q3_freq)
        .map(|(kmer, count)| (*kmer, *count))
        .collect();
    filtered_kmers.sort_by(|a, b| b.1.cmp(&a.1));
    filtered_kmers.into_iter().take(n).map(|(kmer, _)| kmer).collect()
}

fn build_kmer_position_index_sampled(
    contigs: &[(String, Vec<u8>)], k: usize, sampled: &HashSet<u64>
) -> HashMap<u64, Vec<(u32, u32)>> {
    let mut index: HashMap<u64, Vec<(u32, u32)>> = HashMap::new();
    for (contig_idx, (_, seq)) in contigs.iter().enumerate() {
        for (pos, window) in seq.windows(k).enumerate() {
            if let Some(key) = canonical_kmer_u64(window) {
                if sampled.contains(&key) {
                    index.entry(key).or_default().push((contig_idx as u32, pos as u32));
                }
            }
        }
    }
    index
}

fn calculate_interval_histogram_from_index_u64(
    kmer_u64: u64, kmer_index: &HashMap<u64, Vec<(u32, u32)>>, n_contigs: usize, bins: &[usize]
) -> Array1<f64> {
    let n_bins = bins.len() - 1;
    let mut histogram = Array1::<f64>::zeros(n_contigs * n_bins);
    if let Some(positions) = kmer_index.get(&kmer_u64) {
        let mut contig_positions: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(contig_idx_u32, pos_u32) in positions {
            contig_positions.entry(contig_idx_u32 as usize).or_default().push(pos_u32 as usize);
        }
        for (contig_idx, mut pos_list) in contig_positions {
            pos_list.sort_unstable();
            for (a, b) in pos_list.iter().tuple_windows() {
                let interval = b - a;
                for bin_idx in 0..n_bins {
                    if interval >= bins[bin_idx] && interval < bins[bin_idx + 1] {
                        histogram[contig_idx * n_bins + bin_idx] += 1.0;
                        break;
                    }
                }
            }
        }
    }
    for contig_idx in 0..n_contigs {
        let start = contig_idx * n_bins;
        let end = (contig_idx + 1) * n_bins;
        let sum: f64 = histogram.slice(ndarray::s![start..end]).sum();
        if sum > 0.0 {
            histogram.slice_mut(ndarray::s![start..end]).mapv_inplace(|x| x / sum);
        }
    }
    histogram
}

fn build_cosine_distance_matrix(
    selected_kmers: &[u64], kmer_index: &HashMap<u64, Vec<(u32, u32)>>, n_contigs: usize, bins: &[usize]
) -> Array2<f64> {
    let n_bins = bins.len() - 1;
    let n_features = selected_kmers.len() * n_bins;
    let mut feature_matrix = Array2::<f64>::zeros((n_contigs, n_features));
    for (k_idx, &kmer_u64) in selected_kmers.iter().enumerate() {
        let norm_hist = calculate_interval_histogram_from_index_u64(kmer_u64, kmer_index, n_contigs, bins);
        for contig_idx in 0..n_contigs {
            let hist_slice = norm_hist.slice(ndarray::s![contig_idx * n_bins..(contig_idx + 1) * n_bins]);
            feature_matrix.slice_mut(ndarray::s![contig_idx, k_idx * n_bins..(k_idx + 1) * n_bins]).assign(&hist_slice);
        }
    }
    let mut dist_matrix = Array2::<f64>::zeros((n_contigs, n_contigs));
    for i in 0..n_contigs {
        for j in (i + 1)..n_contigs {
            let vec_i = feature_matrix.row(i);
            let vec_j = feature_matrix.row(j);
            let dot_product = vec_i.dot(&vec_j);
            let norm_i = vec_i.dot(&vec_i).sqrt();
            let norm_j = vec_j.dot(&vec_j).sqrt();
            let sim = if norm_i * norm_j > 1e-9 { dot_product / (norm_i * norm_j) } else { 0.0 };
            dist_matrix[[i, j]] = 1.0 - sim;
            dist_matrix[[j, i]] = 1.0 - sim;
        }
    }
    dist_matrix
}

fn auto_detect_dbscan_params(dist_matrix: &Array2<f64>) -> (f64, usize) {
    let n = dist_matrix.nrows();
    let mut knn_distances: Vec<Vec<f64>> = Vec::new();
    for i in 0..n {
        let mut distances: Vec<f64> = (0..n).filter(|&j| i != j).map(|j| dist_matrix[[i, j]]).collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        knn_distances.push(distances);
    }
    
    let max_k = (n - 1).max(1);
    let avg_knn_dist: Vec<f64> = (1..=max_k).map(|k| {
        knn_distances.iter().map(|dists| dists.get(k - 1).copied().unwrap_or(0.0)).sum::<f64>() / n as f64
    }).collect();
    
    let elbow_k = find_elbow_index(&avg_knn_dist) + 1;
    let min_points = elbow_k;
    eprintln!("  k-NN elbow point detected at k={}", elbow_k);
    eprintln!("  Using elbow-based min_points: {}", min_points);
    
    let k_for_eps = min_points.min(max_k);
    let mut k_distances: Vec<f64> = knn_distances.iter()
        .filter_map(|dists| dists.get(k_for_eps - 1).copied())
        .collect();
    k_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    if k_distances.is_empty() {
        eprintln!("  Warning: No valid k-distances found, using fallback eps=0.1");
        return (0.1, min_points);
    }
    
    let knee_idx = find_elbow_index(&k_distances);
    let eps = k_distances[knee_idx];
    eprintln!("  k-distance knee point (k={}): {:.2}", k_for_eps, eps);
    eprintln!("  Selected eps={:.2}, min_points={}", eps, min_points);
    
    (eps, min_points)
}

fn find_elbow_index(values: &[f64]) -> usize {
    if values.len() < 3 {
        return if values.is_empty() { 0 } else { values.len() - 1 };
    }
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    if range.abs() < 1e-10 {
        return values.len() - 1;
    }
    let normalized: Vec<f64> = values.iter().map(|&v| (v - min_val) / range).collect();
    let mut max_curvature = 0.0;
    let mut elbow_idx = 0;
    for i in 1..(normalized.len() - 1) {
        let curvature = (normalized[i + 1] + normalized[i - 1] - 2.0 * normalized[i]).abs();
        if curvature > max_curvature {
            max_curvature = curvature;
            elbow_idx = i;
        }
    }
    elbow_idx
}
