use clap::Parser;
use router_core::NeuralRouter;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the ONNX model
    #[arg(short, long)]
    model: PathBuf,

    /// Path to the tokenizer JSON
    #[arg(short, long)]
    tokenizer: PathBuf,

    /// Text to route
    #[arg(long, default_value = "Write a python script to sort a list")]
    text: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    println!("Loading model from {:?}", cli.model);
    println!("Loading tokenizer from {:?}", cli.tokenizer);
    let router = NeuralRouter::new(&cli.model, &cli.tokenizer)?;

    println!("Routing text: {:?}", cli.text);
    let start = Instant::now();
    let result = router.route(&cli.text)?;
    let duration = start.elapsed();

    println!("Result: {:?}", result);
    println!("Time taken: {:?}", duration);

    Ok(())
}
