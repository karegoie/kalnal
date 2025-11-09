use std::{env, process};

fn main() {
    let config = kalnal::config::Config::new(env::args()).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {}", err);
        eprintln!("Usage: kalnal <kmer> <input_dir>");
        eprintln!("  Performs k-mer counting, clustering, and plotting on all .split.fa files in the directory");
        process::exit(1);
    });

    if let Err(e) = kalnal::startup::run(&config.input_dir, config.k) {
        eprintln!("Application error: {}", e);
        process::exit(1);
    }
}
