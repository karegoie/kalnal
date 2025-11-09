# kalnal
---
The polyploid subgenome divider

A high-performance Rust implementation for k-mer based genomic analysis, hierarchical clustering, and dendrogram visualization.

## Features

- **Fast k-mer counting**: Parallel k-mer counting using rayon
- **Hierarchical clustering**: Ward's method clustering using the `kodama` library  
- **Dendrogram visualization**: Custom dendrogram plots with `plotters`
- **Multiple sample support**: Process and cluster multiple samples in batch mode

## Quick Start

### Building

```bash
cd kalnal-kmer
cargo build --release
```

The binary will be at `kalnal-kmer/target/release/kalnal`.

### Basic Usage

#### Mode 1: K-mer Counting

Count k-mers in a single FASTA file:

```bash
kalnal 21 input.fasta > output.tsv
```

#### Mode 2: Complete Analysis Pipeline

Perform k-mer counting, clustering, and visualization for multiple samples:

```bash
kalnal analyze 21 temp/
```

**Requirements:**
- Input directory must contain `.split.fa` files (one per sample)

**Output:**
- `21_analyzed.png` - Dendrogram visualization
- `K21.combine.json` - Complete k-mer count data

## Documentation

See [kalnal-kmer/USAGE.md](kalnal-kmer/USAGE.md) for detailed documentation.

## Technical Details

- **Language**: Rust
- **Clustering**: Ward's method via `kodama`
- **Visualization**: Custom implementation with `plotters`
- **Performance**: Parallel processing with rayon
- **Memory**: Efficient bit-packed k-mer representation

## Migration from Python

This project has been fully migrated from Python to Rust for:
- Better performance (10-100x faster)
- Lower memory usage
- Single binary distribution (no dependency management)
- Type safety and reliability

## License

See [LICENSE](LICENSE)
