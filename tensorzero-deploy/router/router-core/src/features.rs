use tract_onnx::prelude::Tensor;

pub struct Features {
    pub input_ids: Vec<u32>,
}

impl Features {
    pub fn new(input_ids: Vec<u32>) -> Self {
        Self { input_ids }
    }

    pub fn to_tensor(&self) -> Tensor {
        // Shape: [1, seq_len]
        let shape = [1, self.input_ids.len()];
        // Convert u32 to i64 because ONNX usually expects i64 for indices
        let data: Vec<i64> = self.input_ids.iter().map(|&x| x as i64).collect();
        Tensor::from_shape(&shape, &data).unwrap()
    }
}
