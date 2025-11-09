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

/// Compute pairwise Euclidean distances between samples
fn compute_distance_matrix(data: &Array2<f64>) -> Array1<f64> {
    let n = data.nrows();
    let n_distances = (n * (n - 1)) / 2;
    let mut distances = Array1::zeros(n_distances);
    
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let row_i = data.row(i);
            let row_j = data.row(j);
            
            // Euclidean distance
            let dist: f64 = row_i
                .iter()
                .zip(row_j.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            
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
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let distances = compute_distance_matrix(&data);
        
        // Distance between (0,0) and (1,1) should be sqrt(2)
        assert!((distances[0] - 2_f64.sqrt()).abs() < 1e-10);
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
