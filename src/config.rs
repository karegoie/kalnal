use std::{env, error::Error};

/// Parsing command line k-size and input directory arguments
pub struct Config {
    pub k: usize,
    pub input_dir: String,
}

impl Config {
    pub fn new(mut args: env::Args) -> Result<Config, Box<dyn Error>> {
        let k: usize = match args.nth(1) {
            Some(arg) => match arg.parse() {
                Ok(k) if k > 0 && k < 33 => k,
                Ok(_) => return Err("k-mer length needs to be larger than zero and no more than 32".into()),
                Err(_) => return Err(format!("issue with k-mer length argument: {}", arg).into()),
            },
            None => return Err("k-mer length input required".into()),
        };

        let input_dir = match args.next() {
            Some(arg) => arg,
            None => return Err("input directory argument needed".into()),
        };

        Ok(Config { k, input_dir })
    }
}
