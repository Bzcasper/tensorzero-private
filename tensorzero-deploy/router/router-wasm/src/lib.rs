use router_core::{Features, Model as CoreModel};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Model {
    inner: CoreModel,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<Model, JsValue> {
        console_error_panic_hook::set_once();
        let inner =
            CoreModel::load_from_bytes(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Model { inner })
    }

    pub fn route(&self, input: &[f32]) -> Result<Vec<f32>, JsValue> {
        let input_ids: Vec<u32> = input.iter().map(|&x| x as u32).collect();
        let features = Features::new(input_ids);
        self.inner
            .predict(&features)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
