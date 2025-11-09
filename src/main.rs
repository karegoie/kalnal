use std::{env, process};

fn main() {
    let config = kalnal::config::Config::new(env::args()).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {}", err);
        eprintln!("Usage: kalnal <kmer> <fasta_file>");
        eprintln!("  Performs k-mer counting, clustering, and plotting on all records in the FASTA file");
        process::exit(1);
    });

    if let Err(e) = kalnal::startup::run(&config.fasta_path, config.k) {
        eprintln!("Application error: {}", e);
        process::exit(1);
    }
}
