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
