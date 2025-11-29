use criterion::{criterion_group, criterion_main, Criterion};
use router_core::NeuralRouter;
use std::path::PathBuf;

fn benchmark_routing(c: &mut Criterion) {
    // This expects a dummy model to be present.
    // For now, we'll skip if the model doesn't exist or use a placeholder.
    let model_path = PathBuf::from("../router.onnx");
    let tokenizer_path = PathBuf::from("../tokenizer.json");
    println!("Current directory: {:?}", std::env::current_dir());
    println!("Looking for model at: {:?}", model_path.canonicalize());

    if !model_path.exists() || !tokenizer_path.exists() {
        println!("Skipping benchmark: router.onnx or tokenizer.json not found");
        return;
    }

    let router = NeuralRouter::new(&model_path, &tokenizer_path).unwrap();
    let text = "Write a python script to sort a list";

    c.bench_function("route", |b| b.iter(|| router.route(text).unwrap()));
}

criterion_group!(benches, benchmark_routing);
criterion_main!(benches);
