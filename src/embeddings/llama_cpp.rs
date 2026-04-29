use arrow_array::{builder::Float32Builder, cast::AsArray, ArrayRef, Float32Array};
use arrow_schema::DataType;
use eyre::{Context, ContextCompat};
use super::common::centroid;

#[derive(serde::Deserialize)]
struct EmbedResult {
    pub _index: Option<usize>,
    pub embedding: Vec<Vec<f32>>,
}

#[derive(Debug)]
pub struct LlamaCppInference {
    pub base_url: url::Url,
    pub client: reqwest::Client,
    pub dim: usize,
}

impl LlamaCppInference {
    #[tracing::instrument(skip(self, source))]
    pub fn compute_inner(&self, source: ArrayRef) -> eyre::Result<Float32Array> {
        tracing::trace!(
            len = source.len(),
            nullable = source.is_nullable(),
            "compute_inner called"
        );

        if source.is_nullable() {
            eyre::bail!("Expected non-nullable data type")
        }

        if !matches!(source.data_type(), DataType::Binary) {
            eyre::bail!("Expected Binary data type")
        };

        if source.len() == 0 {
            tracing::debug!(
                "Empty source array, returning empty embeddings (schema inference probe)"
            );
            return Ok(Float32Array::from(Vec::<f32>::new()));
        }

        tracing::debug!(n_images = source.len(), "Encoding images for embedding");

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

        tracing::trace!(
            n_images = inputs.len(),
            "Images encoded as base64, preparing payloads"
        );

        let payloads = inputs
            .into_iter()
            .map(|s| {
                serde_json::json!({
                    "content": "Image: [img-1]",
                    "image_data": [
                        {
                            "id": 1,
                            "data": s,
                        }
                    ]
                })
            })
            .collect::<Vec<_>>();

        let client = self.client.clone();
        let url = self
            .base_url
            .join("/embedding")
            .wrap_err("Failed to join base_url with '/embedding'")?;

        tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(async move {
                use futures::StreamExt;

                let mut stream = futures::stream::iter(payloads)
                    .map(|p| {
                        let client = client.clone();
                        let url = url.clone();
                        async move {
                            let mut attempts = 0;
                            let max_attempts = 3;
                            let res: Vec<EmbedResult> = loop {
                                match client.post(url.clone()).json(&p).send().await {
                                    Ok(resp) => {
                                        match resp.error_for_status() {
                                            Ok(resp) => break resp.json().await.map_err(|e| eyre::eyre!("failed to parse llama-server response: {e}")),
                                            Err(e) if attempts < max_attempts => {
                                                attempts += 1;
                                                tracing::warn!(error = %e, attempt = attempts, "llama-server error, retrying...");
                                                tokio::time::sleep(std::time::Duration::from_secs(attempts)).await;
                                            }
                                            Err(e) => break Err(eyre::eyre!("llama-server returned error: {e}")),
                                        }
                                    }
                                    Err(e) if attempts < max_attempts => {
                                        attempts += 1;
                                        tracing::warn!(error = %e, attempt = attempts, "llama-server request failed, retrying...");
                                        tokio::time::sleep(std::time::Duration::from_secs(attempts)).await;
                                    }
                                    Err(e) => break Err(eyre::eyre!("llama-server request failed: {e}")),
                                }
                            }?;

                            let embed_res =
                                res.get(0).wrap_err("Failed to get embedding result")?;
                            centroid(embed_res.embedding.clone())
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
