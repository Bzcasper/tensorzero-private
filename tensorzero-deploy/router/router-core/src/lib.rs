mod features;
mod model;
mod tokenizer;

pub use features::Features;
pub use model::Model;
pub use tokenizer::Tokenizer;

use anyhow::Result;
use std::path::Path;

pub struct NeuralRouter {
    model: Model,
    tokenizer: Tokenizer,
}

impl NeuralRouter {
    pub fn new<P: AsRef<Path>>(model_path: P, tokenizer_path: P) -> Result<Self> {
        let model = Model::load(model_path)?;
        let tokenizer = Tokenizer::load(tokenizer_path)?;
        Ok(Self { model, tokenizer })
    }

    pub fn route(&self, text: &str) -> Result<Vec<f32>> {
        let input_ids = self.tokenizer.encode(text)?;
        let features = Features::new(input_ids);
        self.model.predict(&features)
    }
}
