use std::{env, process};

fn main() {
    let config = kalnal_kmer::config::Config::new(env::args()).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {}", err);
        process::exit(1);
    });

    if let Err(e) = kalnal_kmer::startup::run(config.path, config.k) {
        eprintln!("Application error: {}", e);
        drop(e);
        process::exit(1);
    }
}
