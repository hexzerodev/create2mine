use std::{env, process};

use create2mine::Config;

fn main() {
    let config = Config::new(env::args()).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {}", err);
        process::exit(1);
    });

    // create2mine::run(config);
    create2mine::gpu(config).unwrap_or_else(|err| {
        eprintln!("Failed to mine: {}", err);
        process::exit(1);
    });
}
