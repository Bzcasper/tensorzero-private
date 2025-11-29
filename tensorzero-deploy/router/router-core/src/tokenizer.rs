use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer as HuggingFaceTokenizer;

pub struct Tokenizer {
    inner: HuggingFaceTokenizer,
}

impl Tokenizer {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = HuggingFaceTokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e))?;
        Ok(Self { inner })
    }

    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self> {
        let inner = HuggingFaceTokenizer::from_bytes(bytes).map_err(|e| anyhow::anyhow!(e))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(encoding.get_ids().to_vec())
    }
}
