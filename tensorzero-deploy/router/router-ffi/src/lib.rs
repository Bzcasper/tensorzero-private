use napi_derive::napi;
use router_core::NeuralRouter as CoreRouter;

#[napi]
pub struct NeuralRouter {
    inner: CoreRouter,
}

#[napi]
impl NeuralRouter {
    #[napi(constructor)]
    pub fn new(model_path: String, tokenizer_path: String) -> napi::Result<Self> {
        let inner = CoreRouter::new(&model_path, &tokenizer_path)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(Self { inner })
    }

    #[napi]
    pub fn route(&self, text: String) -> napi::Result<Vec<f64>> {
        let result = self
            .inner
            .route(&text)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        // Convert f32 to f64 for JS
        Ok(result.into_iter().map(|x| x as f64).collect())
    }
}
