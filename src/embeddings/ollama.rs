use super::common::{centroid, Embedder};
use arrow_array::{builder::Float32Builder, cast::AsArray, ArrayRef, Float32Array};
use arrow_schema::DataType;
use eyre::{Context, ContextCompat};

#[derive(Debug)]
pub struct OllamaInference {
    pub base_url: url::Url,
    pub client: reqwest::Client,
    pub model: String,
    pub dim: usize,
}

#[derive(serde::Deserialize)]
struct OllamaEmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
}

impl Embedder for OllamaInference {
    fn dim(&self) -> usize {
        self.dim
    }

    #[tracing::instrument(skip(self, source))]
    fn embed_array(&self, source: ArrayRef) -> eyre::Result<Float32Array> {
        if source.is_nullable() {
            eyre::bail!("Expected non-nullable data type")
        }
        if !matches!(source.data_type(), DataType::Binary) {
            eyre::bail!("Expected Binary data type")
        };
        if source.len() == 0 {
            return Ok(Float32Array::from(Vec::<f32>::new()));
        }

        let inputs = source
            .as_binary::<i32>()
            .into_iter()
            .map(|b| {
                use base64::{engine::general_purpose, Engine as _};
                let bytes = b.wrap_err("we already asserted that the array is non-nullable")?;
                let b64 = general_purpose::STANDARD.encode(bytes);
                eyre::Ok(b64)
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        let client = self.client.clone();
        let url = self
            .base_url
            .join("/api/embed")
            .wrap_err("Failed to join base_url with '/api/embed'")?;

        tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(async move {
                use futures::StreamExt;
                let mut stream = futures::stream::iter(inputs)
                    .map(|img_b64| {
                        let client = client.clone();
                        let url = url.clone();
                        let model = self.model.clone();
                        async move {
                            // Ollama supports multimodal embeddings if the model supports it.
                            let payload = serde_json::json!({
                                "model": model,
                                "input": "describe this image",
                                "images": [img_b64]
                            });

                            let res: OllamaEmbedResponse = client
                                .post(url)
                                .json(&payload)
                                .send()
                                .await
                                .map_err(|e| eyre::eyre!("ollama request failed: {e}"))?
                                .error_for_status()
                                .map_err(|e| eyre::eyre!("ollama returned error: {e}"))?
                                .json()
                                .await
                                .map_err(|e| eyre::eyre!("failed to parse ollama response: {e}"))?;

                            centroid(res.embeddings)
                        }
                    })
                    .buffer_unordered(4);

                let mut vecres = vec![];
                while let Some(res) = stream.next().await {
                    vecres.push(res?);
                }

                let mut builder = Float32Builder::new();
                for res in vecres {
                    builder.append_slice(&res);
                }
                Ok(builder.finish())
            })
        })
    }
}
