use anyhow::{Context, Result};
use std::path::Path;
use tract_onnx::prelude::*;

type TractPlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct Model {
    runnable: TractPlan,
}

impl Model {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self { runnable: model })
    }

    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = std::io::Cursor::new(bytes);
        let model = tract_onnx::onnx()
            .model_for_read(&mut reader)?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self { runnable: model })
    }

    pub fn predict(&self, features: &crate::Features) -> Result<Vec<f32>> {
        let input = features.to_tensor();
        let result = self.runnable.run(tvec!(input.into()))?;

        let output = result[0]
            .to_array_view::<f32>()?
            .as_slice()
            .context("Failed to convert output to slice")?
            .to_vec();

        Ok(output)
    }
}
